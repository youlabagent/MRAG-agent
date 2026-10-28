"""The CORGI arm of the Section-13 axis validation: runner, GPU lease, expression context, status.

Productionised from /BIO1/MRAG/corgi/ft/ag/corgi_run.py and ft/prad/corgi_run_prad.py (the two
research copies differ only in which expr_contexts.txt they read), with ft/seko.py's predict() for
the test-time augmentation and ft/ag/agstats.py for the read-out track set.

WHAT IT PRODUCES.  The same JSONL as the AlphaGenome arm: one row per prediction,
{"label","tf","kind","sums"} with sums[track][window] = the signal summed over the 64-bp bins the
named read-out window covers, plus a <out>.tracks.json sidecar naming each track's channels and
resolution.  mrag_axis.stat.stats.load()/effect() read exactly that shape and key the rows on
(label, tf, kind), so a resumed or re-run arm is byte-comparable with the stored research results
(/BIO1/MRAG/corgi/ft/prad/corgi_A_prad.jsonl).  The run is resumable: a (label, tf, kind) already
in the output file is never predicted again, which is what makes a 3,050-row arm survive a
preempted GPU.

THREE THINGS IT DOES THAT THE RESEARCH DID NOT, each because a shared production machine demands it

1. THE GPU COMES FROM THE DEPLOYED LEASE, NOT FROM torch.  The research set
   torch.cuda.set_per_process_memory_fraction(0.5, 0) -- about 24.5 GiB of a 49,140-MiB A6000 --
   which leaves 24 GiB and so fails the target scorer's own 32,000-MiB free check
   (ensure_scorer_ollama.sh:40, measured in mrag_orchestrator/gpu_lease.py's header on
   2026-10-18): the scorer then reports "no GPU / RAM free now, retry later" and target scoring
   stops.  The cap therefore comes from gpu_lease.lease(need_gib=...) -- 7.4 GiB against Corgi's
   measured 4.52-GiB peak -- and reaches the child as CUDA_VISIBLE_DEVICES +
   MRAG_GPU_MEMORY_FRACTION, which the child turns into
   torch.cuda.set_per_process_memory_fraction(fraction, 0).  No card means status "skipped" with
   the lease's own .reason: the model-arm failure policy applied
   to the GPU.  NOTHING HERE RAISES on a missing input.

2. THE EXPRESSION CONTEXT IS RESOLVED, NEVER DEFAULTED.  Reference/corgi/expr_{contexts.txt,
   vectors.npy} is a FIXED 31-row table (26 TCGA-PRAD tumours, LNCaP_CT/OE, H1963_LacZ/cMYC and
   PRAD_AR_high_mean, the mean of the 12 AR_high TCGA samples).  The research hardcoded
   --context LNCaP_OE for the OC2 run and PRAD_AR_high_mean for the PRAD one.  resolve_context()
   derives both from the run itself (mrag_axis.stat.config.expr_context + production's own lineage matcher)
   and a run that matches no row ends as status "no context" with the reason in the record --
   conditioning an arbitrary tumour's sequence on a foreign tumour's expression vector would be a
   different experiment, not a fallback (SECTION13_CONTRACT.md, "Open, to decide while
   implementing").

3. CORGI IS hg38-ONLY, so a run whose recorded genome is not hg38 ends as status
   "unsupported (hg38 only)" and the consensus cannot exist (contract, same section).  The genome
   is the one the RUN itself recorded in its manifest, so this arm follows the same organism
   decision every other genome argument of that run followed (mrag_orchestrator.organism.decide).

WHAT IT DOES NOT DO.  It does not build edit items (mrag_axis.prep does, and the AlphaGenome arm replays
the same ones), it does not score anything (s13.stats does), and it imports nothing from
/BIO1/MRAG/corgi/ft: the deployed runtime is Agent/libs/corgi_client/{corgi,.venv} plus
Agent/Reference/corgi (ASSETS.json records the md5 of every asset).

ENTRY POINTS.  run(edits, out, run_dir) is the whole arm and returns the status record -- the
counterpart of mrag_axis.arm.alphagenome.run_layer(edits, out), which takes no run because AlphaGenome
has no expression context, no checkpoint and no card to lease.  probe() is the CPU preflight.
--worker is the GPU half, spawned by run(); it is not called directly.

THE PARENT / CHILD SPLIT.  The node's own interpreter has no torch, and the Corgi venv has no s13,
so the parent (this module imported) resolves the context, the genome and the lease and then runs
THIS SAME FILE as a script under /BIO1/MRAG/Agent/libs/corgi_client/.venv/bin/python with a job
JSON that carries the channel map and the window geometry.  The child therefore imports only
torch / numpy / pyfaidx / corgi, and the parent recounts the output file itself rather than
trusting the child's own report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

# The deployed Corgi runtime (stage 2b, ASSETS.json: installed 20261018_052845).  Nothing here
# reads /BIO1/MRAG/corgi, so the node works on a machine that only has the Agent tree.
CORGI_PY = "/BIO1/MRAG/Agent/libs/corgi_client/.venv/bin/python"
CORGI_REPO = "/BIO1/MRAG/Agent/libs/corgi_client/corgi"
CORGI_CONSTANTS = f"{CORGI_REPO}/corgi/constants.py"
REF = "/BIO1/MRAG/Agent/Reference/corgi"
CKPT = f"{REF}/corgi_model.pt"
EXPR_CONTEXTS = f"{REF}/expr_contexts.txt"
EXPR_VECTORS = f"{REF}/expr_vectors.npy"
HG38_FA = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/hg38.fa"
AGENT_ROOT = "/BIO1/MRAG/Agent"

# gpu_lease.DEFAULT_CAP_GIB.  Repeated as a fallback only: lease_gpu() prefers the deployed
# module's own constant, so a change there moves this arm with it.
NEED_GIB = 7.4

# The child's watchdog.  Measured: 1.17 s per prediction over the 3,050 predictions of the
# published PRAD layer (ft/prad/corgi_A.log), i.e. ~60 min; the default below allows 5 s each
# plus 15 min of start-up (the checkpoint load and the first CUDA context), so a layer that is
# merely slow is never killed while a HUNG child -- a CUDA init that never returns, a card another
# process is thrashing -- costs that much and then becomes a WARN gate instead of a run that waits
# for ever.  This step is synchronous inside the pipeline, which is why a default exists at all.
TIMEOUT_BASE_S = 900.0
TIMEOUT_PER_PREDICTION_S = 5.0


def default_timeout_s(items):
    """The watchdog deadline for a layer of `items`, from its own prediction count."""
    return TIMEOUT_BASE_S + TIMEOUT_PER_PREDICTION_S * max(1, len(expected_keys(items)))

# The read-out track set, from ft/ag/agstats.py's Corgi side and ft/ag/corgi_run.py's TRACKS: the
# eight assays the axis statistic reads (mrag_axis.stat.stats.TGT_OC2/TGT_PRAD name corgi|rna_total,
# corgi|h3k4me3 and corgi|dnase; ENH_RD names corgi|h3k27ac and corgi|dnase).  The CHANNEL INDICES
# are not written down here -- tracks() derives them from the checkpoint's own track-name order
# through mrag_axis.stat.config.corgi_channels(), and test_arm_corgi.py checks the result against the
# research literal {dnase:[0], atac:[1], h3k4me1:[2], h3k4me3:[4], h3k27ac:[7], cage:[12,13],
# rna_total:[16,17], rna_polya:[18,19]}.
READOUTS = ("dnase", "atac", "h3k4me1", "h3k4me3", "h3k27ac", "cage", "rna_total", "rna_polya")
KEY = "corgi|%s"

# The status vocabulary.  SECTION13_CONTRACT.md ("What the node has to emit") requires per-arm
# status {ok | no signal | failed | partial, error}; the other three are the gates this arm adds.
OK, NO_SIGNAL, PARTIAL, FAILED = "ok", "no signal", "partial", "failed"
SKIPPED, NO_CONTEXT, UNSUPPORTED = "skipped", "no context", "unsupported (hg38 only)"


def log_to(sink):
    """ft/ag/corgi_run.py's log(): timestamped, flushed, so a 55-s-per-item arm shows progress."""
    def _log(m):
        if sink is not None:
            sink(time.strftime("[%H:%M:%S] ") + m)
    return _log


def _config():
    """mrag_axis.stat.config, imported on FIRST USE and not at module import.

    The child runs under the Corgi venv, where the build tree (and therefore s13) is not on the
    path: it gets the channel map and the geometry from its job file instead.  A module-level
    `from mrag_axis.stat import config` would make this file unimportable there -- and running it there is
    the whole point of the parent/child split.
    """
    from mrag_axis.stat import config as C
    return C


# ---------------------------------------------------------------- channels and geometry
def tracks(constants=CORGI_CONSTANTS):
    """(TRACKS, geometry, missing): {"corgi|<assay>": [channels]} and WINDOW/FLANK/NBIN/BIN.

    Derived, not copied: mrag_axis.stat.config.corgi_channels() reads corgi.constants.DEFAULT_TRACK_NAMES --
    the checkpoint's own assay order -- and pairs '<name>_plus'/'<name>_minus' into the two-channel
    stranded read-outs (cage, rna_total, rna_polya).  `missing` names any read-out the installed
    constants do not have, which is a stop condition rather than a silently shorter track set.
    """
    d = _config().corgi_channels(constants=constants)
    trk = {KEY % n: list(d["TRACKS"][n]) for n in READOUTS if n in d["TRACKS"]}
    geom = {k: int(d[k]) for k in ("WINDOW", "HALF", "FLANK", "NBIN", "BIN", "n_channels")}
    geom["STRAND_PAIRS"] = [list(p) for p in d["STRAND_PAIRS"]]
    return trk, geom, [n for n in READOUTS if n not in d["TRACKS"]]


def apply_edits(wt, edits):
    """ft/ag/corgi_run.py verbatim: the prep's (position, bases) list applied to the window."""
    s = list(wt)
    for pos, bases in edits:
        s[pos:pos + len(bases)] = bases
    return "".join(s)


def window_sums(P, windows, ws, trk, geom):
    """{track: {window: summed signal}} for one prediction of the window starting at ws.

    ft/ag/corgi_run.py's reduction, kept arithmetic-for-arithmetic: Corgi predicts the CENTRAL
    393,216 bp of its 524,288-bp input, so bin 0 is at ws + FLANK and a read-out interval is
    clipped to [0, NBIN) -- an interval entirely outside the predicted span contributes 0.0, which
    is what the stored artefacts carry and therefore what the statistic was calibrated on.

    Two deliberate differences, neither of which can change a stored value:
    * the per-channel values are summed as ABSOLUTE values, the convention ft/ag/agstats.py
      applies to stranded assays so a negative minus strand cannot cancel the plus strand.  It is
      a measured no-op here: this checkpoint has no output_head, so predict() applies softplus and
      every value is positive -- all 48,800 sums in ft/prad/corgi_A_prad.jsonl are >= 0.0042.
    * a non-finite sum becomes None instead of NaN, because json.dumps writes a bare NaN that no
      strict JSON reader will load back -- a dead model must not also corrupt the output file.
    """
    o0 = ws + geom["FLANK"]
    nbin, bb = geom["NBIN"], geom["BIN"]
    res = {}
    for name, chans in trk.items():
        d = {}
        for wn, ivs in windows.items():
            tot, bad = 0.0, False
            for a, b in ivs:
                i0 = max((a - o0) // bb, 0)
                i1 = min(-(-(b - o0) // bb), nbin)
                if i1 > i0:
                    v = float(np.abs(P[chans, i0:i1]).sum())
                    if math.isfinite(v):
                        tot += v
                    else:
                        bad = True
            d[wn] = None if bad else tot
        res[name] = d
    return res


# ---------------------------------------------------------------- the expression context
# 'PRAD_AR_high_mean' is the mean of the 12 AR_high TCGA RNA samples (ft/prad/corgi_run_prad.py
# header), so '_mean' marks HOW the row was built, not WHICH condition it is; it is the only token
# of the 31 rows that a run can never name.
MARKER_TOKENS = frozenset({"MEAN"})


def _tok(s):
    return [t for t in re.split(r"[^A-Za-z0-9]+", str(s or "").upper()) if t]


def expr_contexts(path=EXPR_CONTEXTS):
    """The fixed context table, in file order -- the row index IS the row of expr_vectors.npy
    (ft/ag/corgi_run.py: ev[ctx.index(args.context)])."""
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


def _lineage_keys(text, agent_root=AGENT_ROOT):
    """The lineage keys production's own matcher assigns to a piece of free text, or () when the
    orchestrator is not importable.  mrag_orchestrator.literature._RULES is the table that says
    'PRAD' and 'prostate cancer' are the same lineage; re-deciding that here would be a second,
    divergent answer to a question production already answers."""
    try:
        p = f"{agent_root}/orchestrator/src"
        if p not in sys.path:
            sys.path.insert(0, p)
        from mrag_orchestrator import literature as L
        return tuple(L.match_item(str(text or "")))
    except Exception:
        return ()


def resolve_context(run_dir=None, *, info=None, lineage=None, contexts=None,
                    contexts_path=EXPR_CONTEXTS, explicit=None, agent_root=AGENT_ROOT):
    """Which row of the fixed 31-row table this run may be conditioned on, or none at all.

    The rule, and why it is a rule rather than a lookup table: a row name is the research
    vocabulary for ONE condition ('LNCaP_OE', 'PRAD_AR_high_mean', 'TCGA-EJ-7318-01'), so a row is
    usable only when the RUN itself names every part of it.  A row qualifies when

      * every token of the row is either a token of the run's own cell line / treatment arm
        (mrag_axis.stat.config.expr_context), or a lineage name that production's matcher maps to this run's
        lineage ('PRAD' -> prostate, which is how the PRAD AR_high run reaches PRAD_AR_high_mean),
        or the '_mean' averaging marker; an unexplained token disqualifies the row, and
      * the run's TREATMENT ARM is named whenever the row needed that lineage step, so a row is
        never chosen on lineage alone.  A row whose every token the run names outright (a TCGA
        barcode spelled out as the run's cell line) needs no arm token.

    The second clause is what keeps the 26 TCGA tumour rows out of every run except one that names
    that barcode: a prostate run must not be conditioned on a stranger's tumour.  Ambiguity is not
    resolved by preference -- two equally specific rows report `name=None` with both names in the
    reason, because guessing would be the same mistake as defaulting.

    Applied to the two research runs this returns exactly the two contexts the research hardcoded
    (test_arm_corgi.py checks both): the OC2 super-enhancer run -> LNCaP_OE,
    the PRAD run -> PRAD_AR_high_mean.
    """
    rows = list(contexts) if contexts is not None else None
    out = {"name": None, "index": None, "source": "", "reason": "", "rna_available": None,
           "run": {}, "matched": [], "candidates": [], "n_contexts": None,
           "contexts_path": contexts_path}
    if rows is None:
        try:
            rows = expr_contexts(contexts_path)
        except Exception as e:
            out["reason"] = f"expression-context table unreadable: {type(e).__name__}: {e}"
            return out
    out["n_contexts"] = len(rows)
    if explicit:
        if explicit in rows:
            out.update(name=explicit, index=rows.index(explicit), source="caller")
        else:
            out["reason"] = (f"context {explicit!r} is not one of the {len(rows)} rows of "
                             f"{contexts_path}")
        return out
    if info is None:
        try:
            info = _config().expr_context(run_dir, agent_root=agent_root)
        except Exception as e:
            out["reason"] = f"run context not derivable: {type(e).__name__}: {e}"
            return out
    out["run"] = dict(info)
    out["rna_available"] = info.get("rna_available")
    arm = set(_tok(info.get("arm")))
    own = arm | set(_tok(info.get("cell_line")))
    if lineage is None:
        # The run lineage comes from stage 2a: mrag_axis.stat.config.lineage() applies production own
        # SUBSTITUTE_BY_LINEAGE keyword table to cell_line + tissue + cancer_type and returns both
        # the label and the text it matched, and literature.match_item() on that same text is what
        # puts PRAD and "prostate cancer" in one lineage.  mrag_axis.stat.config.expr_context() carries only
        # the arm and the cell line, so the tissue has to be read from the run, not from `info`.
        lineage = set()
        try:
            lab, hay = _config().lineage(run_dir, agent_root=agent_root) if run_dir else (None, "")
        except Exception:
            lab, hay = None, ""
        if lab:
            lineage.add(lab)
        lineage |= set(_lineage_keys(hay, agent_root))
        for k in ("cell_line", "tissue", "cancer_type"):   # when the caller injected `info`
            lineage |= set(_lineage_keys(info.get(k), agent_root))
    lineage = {str(x).lower() for x in lineage}
    scored = []
    for i, nm in enumerate(rows):
        toks = _tok(nm)
        hit = [t for t in toks if t in own]
        rest = [t for t in toks if t not in own and t not in MARKER_TOKENS]
        coh = [t for t in rest if lineage & {x.lower() for x in _lineage_keys(t, agent_root)}]
        if len(rest) != len(coh) or (coh and not arm & set(hit)):
            continue
        scored.append((len(hit), -len(coh), i, nm, hit, coh))
    out["candidates"] = [s[3] for s in scored]
    if not scored:
        out["reason"] = (f"no row of {contexts_path} ({len(rows)} rows) is named by this run "
                         f"(cell line {info.get('cell_line')!r}, arm {info.get('arm')!r}, "
                         f"lineage {sorted(lineage) or 'unknown'}); the table is the PRAD and "
                         f"cell-line panel, and a foreign tumour's expression vector is not a "
                         f"substitute")
        return out
    # most tokens matched to the run own vocabulary first, then fewest tokens that needed the
    # lineage step -- i.e. the most specific row, never a preference between two equal ones
    best = max(s[:2] for s in scored)
    top = [s for s in scored if s[:2] == best]
    if len(top) > 1:
        out["reason"] = (f"{len(top)} rows are named equally well by this run "
                         f"({', '.join(s[3] for s in top)}); refusing to guess which condition "
                         f"the run is")
        return out
    s = top[0]
    out.update(name=s[3], index=s[2], source="run", matched=s[4] + s[5])
    return out


# ---------------------------------------------------------------- the organism gate
def genome_of(run_dir=None, *, explicit=None, out=None, agent_root=AGENT_ROOT):
    """{"genome","organism","source","reason"} for the run -- Corgi needs "hg38".  Never raises.

    Precedence, and why: the run's OWN manifest step 'organism' is authoritative, because that is
    the decision every other genome argument of that run followed (mrag_orchestrator/organism.py:
    "One profile now answers every genome question ... `decide()` records WHERE the answer came
    from").  Falling back to organism.decide() on the run's declared GTF reproduces production's
    documented precedence, including its visible human default, instead of inventing a second one
    -- and a mouse run then stops this arm rather than being quietly scored against hg38, the
    exact failure organism.py was written for.
    """
    r = {"genome": None, "organism": None, "source": "", "reason": ""}
    try:
        p = f"{agent_root}/orchestrator/src"
        if p not in sys.path:
            sys.path.insert(0, p)
        from mrag_orchestrator import organism as O
    except Exception as e:
        r["reason"] = f"mrag_orchestrator.organism not importable: {type(e).__name__}: {e}"
        O = None
    if explicit not in (None, ""):
        if O is None:
            return r
        try:
            prof = O.resolve(explicit)
            return {"genome": prof.genome, "organism": prof.key, "source": "caller",
                    "reason": f"caller set organism={explicit!r}"}
        except Exception as e:
            r["reason"] = f"organism={explicit!r} unresolvable: {type(e).__name__}: {e}"
            return r
    try:
        C = _config()
        out = out or C.run_output(run_dir, agent_root=agent_root)
        t = C.run_toml(run_dir)
    except Exception as e:
        r["reason"] = f"run not readable: {type(e).__name__}: {e}"
        return r
    rec = _manifest_step_values(out, "organism")
    if rec.get("genome"):
        return {"genome": rec["genome"], "organism": rec.get("organism"),
                "source": "manifest[organism]", "reason": rec.get("note", "")}
    if O is None:
        return r
    try:
        d = O.decide(explicit=t.get("organism") or None,
                     artifact_paths=[str((t.get("pipeline") or {}).get("gtf_path") or "")],
                     where=str(run_dir))
        return {"genome": d.profile.genome, "organism": d.profile.key,
                "source": f"organism.decide/{d.source}", "reason": d.detail or d.note or ""}
    except Exception as e:
        r["reason"] = f"organism undecidable: {type(e).__name__}: {e}"
        return r


def _manifest_step_values(out, step):
    """{key: value} of one manifest step's value-only outputs.  mrag_axis.stat.config.manifest() keeps only
    outputs that carry a path, and the organism step records values (organism, genome, profile)."""
    p = f"{out}/manifest.json"
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            d = json.load(fh)
    except Exception:
        return {}
    for st in d.get("steps") or []:
        if st.get("name") == step:
            v = {o.get("key"): o.get("value") for o in (st.get("outputs") or []) if o.get("key")}
            v["note"] = st.get("note") or ""
            return v
    return {}


# ---------------------------------------------------------------- the GPU lease
class _NoGpu:
    """What lease_gpu() returns when the deployed lease module cannot even be imported.

    Shaped like gpu_lease._NoLease on purpose -- falsy, carries .reason, env() blanks
    CUDA_VISIBLE_DEVICES so a child cannot inherit a stale card -- so the caller has one code
    path whether the refusal came from the lease or from the import of it.
    """

    index = None
    cap_gib = 0.0
    memory_fraction = 0.0
    per_gpu: dict = {}

    def __init__(self, reason):
        self.reason = reason

    def __bool__(self):
        return False

    def env(self):
        return {"CUDA_VISIBLE_DEVICES": ""}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def __repr__(self):
        return "NoGpu(%s)" % self.reason


def lease_gpu(need_gib=None, lock_dir=None, wait_s=0, log=None, **kw):
    """One card through the DEPLOYED lease (`from mrag_orchestrator import gpu_lease`).

    need_gib defaults to the lease module's own DEFAULT_CAP_GIB (7.4, from Corgi's measured
    4.52-GiB peak), so the cap tracks that module instead of a second copy of the number here.
    A refusal is falsy and carries .reason; an unimportable lease module is the same thing.
    """
    try:
        from mrag_orchestrator import gpu_lease
    except Exception:
        try:
            p = f"{AGENT_ROOT}/orchestrator/src"
            if p not in sys.path:
                sys.path.insert(0, p)
            from mrag_orchestrator import gpu_lease
        except Exception as e:
            return _NoGpu(f"mrag_orchestrator.gpu_lease not importable: {type(e).__name__}: {e}")
    cap = getattr(gpu_lease, "DEFAULT_CAP_GIB", NEED_GIB) if need_gib is None else need_gib
    return gpu_lease.lease(need_gib=cap, lock_dir=lock_dir, wait_s=wait_s, log=log, **kw)


def child_env(lease, base=None):
    """The child's environment: the parent's own, plus EXACTLY what the lease dictates.

    CUDA_VISIBLE_DEVICES=<leased card> is the pinning -- the child then sees one card and calls it
    cuda:0 -- and MRAG_GPU_MEMORY_FRACTION is the cap gpu_lease computed from need_gib
    (7.4 GiB of a 49,140-MiB A6000 = 0.1542).  Neither is a constant in this file: a lease that
    hands out a different card or cap moves the child with it.
    """
    env = dict(os.environ if base is None else base)
    env.update(lease.env())
    env.setdefault("PYTHONUNBUFFERED", "1")     # the child's per-item log must reach the run log live
    return env


def gpu_setup(torch_mod, env):
    """What the CHILD does before it touches the model: pin to the leased card, cap the memory.

    The fraction is read from the environment the lease produced and is never a literal here --
    the research value, torch.cuda.set_per_process_memory_fraction(0.5, 0) in ft/ag/corgi_run.py,
    is about 24.5 GiB and leaves the target scorer 24 GiB against its 32,000-MiB requirement, so
    it would stop target scoring (gpu_lease.py header, measured 2026-10-18).
    Returns {"error": ...} instead of raising, and refuses to run at all without a leased card:
    a silent CPU fallback would turn a 1.17-s prediction into minutes and read as a hung run.
    """
    vis = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not vis:
        return {"error": "no leased card in the environment (CUDA_VISIBLE_DEVICES is empty); the "
                         "parent must hold a gpu_lease before starting this child"}
    raw = env.get("MRAG_GPU_MEMORY_FRACTION")
    try:
        frac = float(raw)
    except (TypeError, ValueError):
        return {"error": f"MRAG_GPU_MEMORY_FRACTION={raw!r} is not a number; it must come from "
                         f"gpu_lease.Lease.memory_fraction"}
    if not 0.0 < frac <= 1.0:
        return {"error": f"MRAG_GPU_MEMORY_FRACTION={frac!r} is outside (0, 1]"}
    if not torch_mod.cuda.is_available():
        return {"error": f"CUDA_VISIBLE_DEVICES={vis} but torch.cuda.is_available() is False"}
    torch_mod.cuda.set_per_process_memory_fraction(frac, 0)
    # The leased card is the ONLY visible one, so cuda:0 is it; the child cannot address another
    # run's card even by accident.
    return {"visible": vis, "device": "cuda:0", "index": 0, "fraction": frac}


# ---------------------------------------------------------------- the item loop
def load_edits(path, geom=None):
    """(items, window, reason).  The edits JSON is mrag_axis.prep's (= ft/ag/agprep_A.py's) output and is
    the SAME file the AlphaGenome arm replays; a window that is not the model's is a stop
    condition, as `assert int(ed["window"]) == WINDOW` was in the research script."""
    try:
        with open(path) as fh:
            ed = json.load(fh)
        items, w = list(ed["items"]), int(ed["window"])
    except Exception as e:
        return None, None, f"edits {path} unreadable: {type(e).__name__}: {e}"
    if geom and w != geom["WINDOW"]:
        return None, w, (f"edits window {w} != the model's {geom['WINDOW']}: the sequences would "
                         f"not be the ones the prep hashed")
    return items, w, ""


def expected_keys(items):
    """[(label, tf, kind)] in the order the research wrote them: the item's WT first, then each
    factor's knockout and its paired control, in the item's own factor order."""
    out = []
    for rec in items:
        out.append((rec["label"], "WT", "wt"))
        for tf in rec["tfs"]:
            out += [(rec["label"], tf, "ko"), (rec["label"], tf, "ctl")]
    return out


def heal_tail(out_path, log=None):
    """Drop a half-written last line before appending to it, and say how many bytes went.

    A run killed between write() and flush() (a preempted GPU, an OOM kill) leaves a truncated
    JSON line.  Appending after it would make the file permanently unparseable -- mrag_axis.stat.stats.load()
    json-loads EVERY line -- so the partial line is cut and that prediction is simply made again.
    """
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return 0
    with open(out_path, "rb+") as fh:
        size = fh.seek(0, os.SEEK_END)
        fh.seek(size - 1)
        if fh.read(1) == b"\n":
            return 0
        n = min(size, 1 << 20)
        fh.seek(size - n)
        i = fh.read(n).rfind(b"\n")
        cut = (size - n + i + 1) if i >= 0 else 0
        fh.truncate(cut)
    if log:
        log(f"dropped a half-written last line of {out_path} ({size - cut} bytes)")
    return size - cut


def read_keys(out_path):
    """{(label, tf, kind)} already in the output -- what makes the arm resumable.  A truncated
    last line (a killed run) is dropped rather than fatal, and is simply predicted again."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as fh:
        for l in fh:
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                done.add((r["label"], r["tf"], r["kind"]))
            except Exception:
                continue
    return done


def verify(out_path, items, trk):
    """What the output file actually contains, recounted from the file itself.

    The PARENT calls this after the child exits, so "ok" is a property of the JSONL on disk and
    not of the child's own report: a child that dies after printing success cannot fake it.
    """
    want = expected_keys(items)
    rows, nulls, bad, seen = 0, 0, [], set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for i, l in enumerate(fh, 1):
                if not l.strip():
                    continue
                rows += 1
                try:
                    r = json.loads(l)
                    k = (r["label"], r["tf"], r["kind"])
                    s = r["sums"]
                except Exception as e:
                    bad.append(f"line {i}: {type(e).__name__}")
                    continue
                seen.add(k)
                if set(s) != set(trk):
                    bad.append(f"line {i}: tracks {sorted(set(s) ^ set(trk))}")
                    continue
                vals = [v for d in s.values() for v in d.values()]
                if vals and all(v is None for v in vals):
                    nulls += 1
    missing = [k for k in want if k not in seen]
    return {"expected": len(want), "rows": rows, "present": len(seen & set(want)),
            "missing": len(missing), "missing_keys": [list(k) for k in missing[:10]],
            "null_rows": nulls, "bad_rows": bad[:10], "extra": len(seen - set(want))}


def classify(v, error=""):
    """(status, reason) from the recount -- one place, so the child and the parent agree.

    An error with rows on disk is "partial", not "failed": the rows are real predictions and the
    next run resumes from them, which is the difference the report has to be able to state
    (SECTION13_CONTRACT.md, "per-arm status ... so the report can say which arm is missing and
    why").  All-null output is "no signal" rather than "ok" because a model that returns nothing
    usable must not be read as evidence of no effect.
    """
    if v["bad_rows"]:
        return FAILED, "output file is not the agreed shape: " + "; ".join(v["bad_rows"])
    if error:
        return (PARTIAL if v["rows"] else FAILED), error
    if not v["expected"]:
        return FAILED, "no edit items to predict"
    if v["missing"]:
        return PARTIAL, (f"{v['missing']} of {v['expected']} predictions missing, first "
                         f"{v['missing_keys'][:3]}")
    # v["extra"] (rows this item list does not name) is recorded but not a failure: a debug run
    # with --limit-items legitimately leaves rows of the full set in the file.
    if v["null_rows"] >= v["rows"] > 0:
        return NO_SIGNAL, (f"all {v['rows']} predictions produced only non-finite window sums; "
                           f"the model ran but returned nothing usable")
    return OK, ""


def run_items(items, out_path, predict_fn, seq_fn, trk, geom, log=None, limit_items=0):
    """Append one row per missing (label, tf, kind), in the research order.  Never raises.

    ft/ag/corgi_run.py's loop, with its two guards kept (an item whose rebuilt sequence does not
    match the prep's md5 is skipped, not predicted, because it is not the sequence the edits were
    computed on) and its crash replaced: the first prediction that raises stops the arm and is
    reported, because a failing model fails for every remaining sequence too -- 3,040 more
    attempts would only delay the report.  Rows are flushed one at a time so a killed run resumes
    at the next prediction.
    """
    say = log_to(log)
    if limit_items:
        items = items[:limit_items]
    c = {"items": len(items), "done_before": 0, "predictions": 0, "written": 0,
         "skipped_items": [], "error": "", "elapsed_s": 0.0}
    sidecar = out_path + ".tracks.json"
    want = {k: {"channels": v, "resolution": geom["BIN"]} for k, v in trk.items()}
    if os.path.exists(sidecar):
        try:
            have = json.load(open(sidecar))
        except Exception:
            have = None
        if have != want:
            # The sidecar is what mrag_axis.stat.stats.effect() reads the channels out of, so appending rows
            # of one track set to a file described by another would be scored as if they matched.
            c["error"] = (f"{sidecar} describes a different track set ({sorted(have or [])}) "
                          f"than this run ({sorted(want)}); resuming would mix two read-out sets")
            return c
    else:
        with open(sidecar, "w") as fh:
            json.dump(want, fh, indent=1)
    heal_tail(out_path, say)
    done = read_keys(out_path)
    c["done_before"] = len(done)
    t0 = time.time()
    say(f"{len(items)} items, {len(done)} predictions already on disk")
    with open(out_path, "a") as fh:
        for i, rec in enumerate(items):
            try:
                wt = seq_fn(rec["chrom"], rec["ws"], geom["WINDOW"])
            except Exception as e:
                c["error"] = f"{rec['label']}: sequence unreadable: {type(e).__name__}: {e}"
                break
            if len(wt) != geom["WINDOW"]:
                c["skipped_items"].append(f"{rec['label']}: window truncated ({len(wt)} bp)")
                continue
            if hashlib.md5(wt.encode()).hexdigest() != rec["wt_md5"]:
                c["skipped_items"].append(f"{rec['label']}: md5 differs from the prep's")
                continue
            jobs = [("WT", "wt", wt)]
            for tf, e in rec["tfs"].items():
                jobs += [(tf, "ko", apply_edits(wt, e["ko"])),
                         (tf, "ctl", apply_edits(wt, e["ctl"]))]
            for tf, kind, seq in jobs:
                if (rec["label"], tf, kind) in done:
                    continue
                try:
                    P = predict_fn(seq)
                except Exception as e:
                    c["error"] = (f"prediction failed at {rec['label']}/{tf}/{kind}: "
                                  f"{type(e).__name__}: {e}")
                    break
                c["predictions"] += 1
                fh.write(json.dumps({"label": rec["label"], "tf": tf, "kind": kind,
                                     "sums": window_sums(P, rec["windows"], rec["ws"], trk,
                                                         geom)}) + "\n")
                fh.flush()
                c["written"] += 1
                done.add((rec["label"], tf, kind))
            if c["error"]:
                break
            say(f"[{i+1}/{len(items)}] {rec['label']}: {len(rec['tfs'])} factors, "
                f"{(time.time()-t0)/max(c['predictions'], 1):.2f}s/prediction")
    c["elapsed_s"] = round(time.time() - t0, 2)
    if c["error"]:
        say("stopped: " + c["error"])
    return c


# ---------------------------------------------------------------- the parent
def _iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def run(edits, out, run_dir=None, *, explicit_context=None, explicit_genome=None, need_gib=None,
        lock_dir=None, limit_items=0, lease_fn=None, child_python=CORGI_PY,
        constants=CORGI_CONSTANTS, checkpoint=CKPT, contexts_path=EXPR_CONTEXTS,
        vectors_path=EXPR_VECTORS, fasta=HG38_FA, timeout_s=None, log=None,
        agent_root=AGENT_ROOT, preflight=True):
    """Run the Corgi arm for one run and return its status record (also written to
    <out>.status.json).  NEVER raises (the model-arm failure policy).

    The gates are ordered so that a run which cannot use a GPU never takes one: the edits, then
    the organism, then the expression context, then the installed runtime (preflight(), which
    loads the checkpoint on the CPU in another process), and only then the lease.  test_arm_corgi.py asserts the lease is not even attempted for a mouse run or a run
    with no context -- holding a card for a step that is about to report "no context" would block
    the target scorer for nothing.
    """
    say = log_to(log)
    t0 = time.time()
    rec = {"arm": "corgi", "status": FAILED, "reason": "", "error": "", "run": run_dir,
           "edits": edits, "out": out, "started": _iso(t0), "ended": "", "elapsed_s": 0.0,
           "genome": {}, "context": {}, "gpu": {}, "runtime": {}, "tracks": {}, "geometry": {},
           "counts": {}, "verify": {}, "child": {}}
    try:
        trk, geom, missing = tracks(constants)
        rec["tracks"], rec["geometry"] = trk, geom
        if missing:
            rec["reason"] = (f"the installed corgi.constants has no track named {missing}; the "
                             f"read-out set of ft/ag/agstats.py cannot be built")
            return _finish(rec, t0, say)
        items, window, why = load_edits(edits, geom)
        if why:
            rec["reason"] = why
            return _finish(rec, t0, say)
        if limit_items:
            items = items[:limit_items]
        rec["counts"]["items"] = len(items)

        rec["genome"] = g = genome_of(run_dir, explicit=explicit_genome, agent_root=agent_root)
        if g.get("genome") != "hg38":
            rec["status"] = UNSUPPORTED
            rec["reason"] = (f"Corgi is trained on hg38 only; this run's genome is "
                             f"{g.get('genome')!r} ({g.get('source') or 'undetermined'}: "
                             f"{g.get('reason')}). Only the AlphaGenome arm can exist, and the "
                             f"two-model consensus cannot.")
            return _finish(rec, t0, say)

        rec["context"] = ctx = resolve_context(run_dir, contexts_path=contexts_path,
                                               explicit=explicit_context, agent_root=agent_root)
        if ctx.get("name") is None:
            rec["status"] = NO_CONTEXT
            rec["reason"] = ctx.get("reason") or "no expression context for this run"
            return _finish(rec, t0, say)

        rec["runtime"] = rt = {"python": child_python, "repo": CORGI_REPO,
                               "checkpoint": checkpoint, "contexts": contexts_path,
                               "vectors": vectors_path, "fasta": fasta}
        gone = [k for k, v in rt.items() if not os.path.exists(v)]
        if gone:
            rec["reason"] = ("the deployed Corgi runtime is incomplete: missing "
                             + ", ".join(f"{k}={rt[k]}" for k in gone))
            return _finish(rec, t0, say)

        if preflight:
            # ~5 s against an arm of 3,050 predictions at 1.17 s each (ft/prad/corgi_A.log): a
            # broken venv or a truncated checkpoint is then a status instead of something a child
            # discovers while it is already holding a card the target scorer wants.
            rec["runtime"]["probe"] = pf = probe(child_python, checkpoint)
            if not pf.get("ok"):
                rec["reason"] = ("the deployed Corgi runtime does not load the checkpoint on the "
                                 "CPU: " + str(pf.get("error"))[:300])
                return _finish(rec, t0, say)

        job = {"edits": edits, "out": out, "limit_items": limit_items, "tracks": trk,
               "geometry": geom, "context": {"name": ctx["name"], "index": ctx["index"]},
               "paths": dict(rt), "result": out + ".worker.json"}
        job_path = out + ".job.json"
        with open(job_path, "w") as fh:
            json.dump(job, fh, indent=1)

        lease = (lease_fn or (lambda: lease_gpu(need_gib=need_gib, lock_dir=lock_dir,
                                                log=say)))()
        with lease as gpu:
            rec["gpu"] = {"leased": bool(gpu), "index": getattr(gpu, "index", None),
                          "cap_gib": getattr(gpu, "cap_gib", None),
                          "memory_fraction": getattr(gpu, "memory_fraction", None),
                          "free_before_mib": getattr(gpu, "free_before_mib", None),
                          "reason": getattr(gpu, "reason", "") or ""}
            if not gpu:
                rec["status"] = SKIPPED
                rec["reason"] = (f"no GPU for the Corgi arm: {rec['gpu']['reason']}. The "
                                 f"AlphaGenome arm is unaffected; re-run to fill this arm in "
                                 f"(the output is resumable).")
                return _finish(rec, t0, say)
            if not timeout_s:
                timeout_s = default_timeout_s(items)
                say(f"no timeout given; the child gets the derived default {timeout_s:.0f}s "
                    f"for {len(expected_keys(items))} prediction(s)")
            rec["timeout_s"] = float(timeout_s)
            rec["child"] = _spawn(child_python, job_path, child_env(gpu), timeout_s, say)
    except Exception as e:                       # the arm reports, it does not raise (policy)
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["reason"] = rec["reason"] or rec["error"]
        return _finish(rec, t0, say)
    try:
        wr = {}
        if os.path.exists(job["result"]):
            with open(job["result"]) as fh:
                wr = json.load(fh)
        rec["counts"] = {**rec["counts"], **(wr.get("counts") or {})}
        rec["verify"] = v = verify(out, items, trk)
        err = (wr.get("counts") or {}).get("error") or wr.get("error") or ""
        if not wr and rec["child"].get("returncode") != 0:
            err = (f"the Corgi child exited {rec['child'].get('returncode')} without a result "
                   f"file: {rec['child'].get('tail', '')[-400:]}")
        rec["status"], rec["reason"] = classify(v, err)
        rec["error"] = err
    except Exception as e:
        rec["status"], rec["error"] = FAILED, f"{type(e).__name__}: {e}"
        rec["reason"] = rec["error"]
    return _finish(rec, t0, say)


def _spawn(child_python, job_path, env, timeout_s, say):
    """Run this file as a script under the Corgi venv, streaming its log into ours.

    Streamed rather than captured because the arm is ~1.17 s per prediction (ft/prad/corgi_A.log)
    and 3,050 predictions: a run log that only shows the child's output after an hour cannot be
    watched.  The last lines are kept in the record so a failure has evidence in the report.

    The deadline is enforced by p.wait(timeout=...) on this thread, with a reader THREAD pumping
    the output, because a check inside the read loop only fires when the child prints: a child
    that hangs silently (a CUDA init that never returns) would never be killed, and this step is
    synchronous inside the run.
    """
    cmd = [child_python, "-P", os.path.abspath(__file__), "--worker", job_path]
    t0, tail = time.time(), []
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)

    def _pump():
        for line in p.stdout:
            line = line.rstrip("\n")
            say("  [corgi] " + line)
            tail.append(line)
            if len(tail) > 40:
                del tail[0]

    th = threading.Thread(target=_pump, daemon=True)
    th.start()
    killed = ""
    try:
        p.wait(timeout=timeout_s if timeout_s else None)
    except subprocess.TimeoutExpired:
        p.kill()
        killed = f"killed after {timeout_s}s (the child stopped before it finished)"
        try:
            p.wait(timeout=30)
        except Exception:                        # noqa: BLE001 -- recorded as killed either way
            pass
    th.join(timeout=10)
    return {"cmd": cmd, "returncode": p.returncode, "elapsed_s": round(time.time() - t0, 2),
            "killed": killed, "tail": "\n".join(tail)}


def _finish(rec, t0, say):
    rec["ended"] = _iso()
    rec["elapsed_s"] = round(time.time() - t0, 2)
    try:
        with open(rec["out"] + ".status.json", "w") as fh:
            json.dump(rec, fh, indent=1)
    except Exception as e:                       # a status file we cannot write is not a failure
        rec["error"] = (rec["error"] + f" | status file unwritable: {type(e).__name__}: {e}").strip()
    say(f"corgi arm: {rec['status']}" + (f" -- {rec['reason']}" if rec["reason"] else ""))
    return rec


# ---------------------------------------------------------------- the child (Corgi venv only)
# ft/seko.py's LUT / onehot / shift_t / rc_pred / predict, copied rather than imported: seko.py
# lives in the research tree, reads research paths at import and needs pandas.  The test-time
# augmentation is part of the measurement, not a nicety -- every stored row in
# ft/prad/corgi_A_prad.jsonl is the mean of 6 passes (3 shifts x 2 strands).
LUT = np.full(256, -1, dtype=np.int16)
for _i, _b in enumerate("ACGT"):
    LUT[ord(_b)] = _i


def onehot(seq):
    idx = LUT[np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)]
    x = np.zeros((len(seq), 4), dtype=np.float32)
    ok = idx >= 0
    x[np.where(ok)[0], idx[ok]] = 1.0          # an N stays all-zero instead of scoring as a base
    return x


def _shift_t(torch, x, s):
    if s == 0:
        return x
    pad = torch.zeros(x.shape[0], abs(s), 4, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-s]], 1) if s > 0 else torch.cat([x[:, -s:], pad], 1)


def _rc_pred(torch, p, pairs):
    p = torch.flip(p, dims=[2]).clone()
    for a, b in pairs:                          # the strands swap with the sequence
        p[:, [a, b]] = p[:, [b, a]]
    return p


def predict(torch, model, seq, v, device, sp, pairs, tta=True):
    """ft/seko.py's predict(): 3 shifts x 2 strands averaged, softplus applied after the mean."""
    with torch.no_grad():
        x = torch.from_numpy(onehot(seq)).unsqueeze(0).to(device)
        outs = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in ((-2, 0, 2) if tta else (0,)):
                xs = _shift_t(torch, x, s)
                outs.append(model(xs, v).float())
                if tta:
                    outs.append(_rc_pred(torch, model(torch.flip(xs, dims=[1])[..., [3, 2, 1, 0]],
                                                      v).float(), pairs))
        o = torch.stack(outs).mean(0)[0]
        return (torch.nn.functional.softplus(o) if sp else o).cpu().numpy()


def build_model(torch, checkpoint, device="cpu"):
    """Load the deployed checkpoint and build the model.  ft/ag/corgi_run.py verbatim, including
    the reason for the softplus flag: a checkpoint WITH an output_head applies softplus inside the
    model, one without it needs softplus applied to the prediction (sp = not final_softplus)."""
    from corgi.config import config_corgi
    from corgi.model import Corgi
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sd = obj["model_state_dict"]
    cfg = dict(config_corgi)
    cfg["final_softplus"] = "output_head.0.weight" in sd
    model = Corgi(cfg)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    sp = not cfg["final_softplus"]
    n = sum(p.numel() for p in model.parameters())
    del obj, sd
    return model, sp, {"final_softplus": cfg["final_softplus"], "softplus_applied_after": sp,
                       "parameters": int(n), "device": str(device)}


def _worker(job_path):
    """The GPU half: pin, cap, load, predict.  Writes its own result JSON and never raises."""
    with open(job_path) as fh:
        job = json.load(fh)
    res = {"status": FAILED, "reason": "", "error": "", "counts": {}, "gpu": {}, "model": {},
           "context": job.get("context", {})}
    try:
        import torch
        res["gpu"] = g = gpu_setup(torch, os.environ)
        if g.get("error"):
            res["error"] = res["reason"] = g["error"]
            return _worker_done(job, res)
        p, geom, trk = job["paths"], job["geometry"], job["tracks"]
        items, _w, why = load_edits(job["edits"], geom)
        if why:
            res["error"] = res["reason"] = why
            return _worker_done(job, res)
        names = expr_contexts(p["contexts"])
        i = int(job["context"]["index"])
        if not (0 <= i < len(names)) or names[i] != job["context"]["name"]:
            res["error"] = res["reason"] = (
                f"expression-context table changed under the run: row {i} of {p['contexts']} is "
                f"{names[i] if 0 <= i < len(names) else None!r}, not "
                f"{job['context']['name']!r}")
            return _worker_done(job, res)
        model, sp, info = build_model(torch, p["checkpoint"], torch.device(g["device"]))
        res["model"] = info
        ev = np.load(p["vectors"]).astype(np.float32)
        v = torch.from_numpy(ev[i]).unsqueeze(0).to(g["device"])
        from pyfaidx import Fasta
        fa = Fasta(p["fasta"])
        pairs = [tuple(x) for x in geom["STRAND_PAIRS"]]
        print(f"[corgi] {info['parameters']} parameters on {g['device']} "
              f"(CUDA_VISIBLE_DEVICES={g['visible']}, memory fraction {g['fraction']}), "
              f"context {job['context']['name']} (row {i} of {len(names)})", flush=True)
        c = run_items(items, job["out"],
                      lambda s: predict(torch, model, s, v, g["device"], sp, pairs),
                      lambda ch, ws, n: str(fa[ch][ws:ws + n].seq).upper(),
                      trk, geom, log=print, limit_items=job.get("limit_items") or 0)
        res["counts"] = c
        res["status"], res["reason"] = classify(verify(job["out"], items, trk), c["error"])
        res["error"] = c["error"]
    except Exception as e:
        import traceback
        res["error"] = f"{type(e).__name__}: {e}"
        res["reason"] = res["error"]
        print(traceback.format_exc()[-2000:], flush=True)
    return _worker_done(job, res)


def _worker_done(job, res):
    try:
        with open(job["result"], "w") as fh:
            json.dump(res, fh, indent=1)
    except Exception as e:
        print(f"[corgi] result file unwritable: {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"[corgi] {res['status']}" + (f" -- {res['reason']}" if res["reason"] else ""),
          flush=True)
    return 0 if res["status"] in (OK, NO_SIGNAL, PARTIAL) else 2


# ---------------------------------------------------------------- preflight (no GPU, no lease)
def _probe_child(checkpoint=CKPT):
    """Prove the installed runtime works WITHOUT a card: import corgi, load the checkpoint to the
    CPU, build the model, load every weight strictly.  Printed as JSON for the parent."""
    r = {"ok": False, "checkpoint": checkpoint, "error": ""}
    try:
        import torch
        r["torch"] = torch.__version__
        import corgi
        r["corgi"] = os.path.dirname(os.path.abspath(corgi.__file__))
        import corgi.constants as K
        r["geometry"] = {"WINDOW": int(K.WINDOW_SIZE), "BIN": int(K.BIN_SIZE),
                         "NBIN": int(K.CENTRAL_BINS), "FLANK": int(K.FLANK),
                         "n_channels": len(K.DEFAULT_TRACK_NAMES)}
        t0 = time.time()
        _m, sp, info = build_model(torch, checkpoint, "cpu")
        r.update(info, softplus_applied_after=sp, load_s=round(time.time() - t0, 2),
                 cuda_available=bool(torch.cuda.is_available()), ok=True)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def probe(child_python=CORGI_PY, checkpoint=CKPT, timeout_s=300):
    """The parent-side preflight: run _probe_child under the Corgi venv and return its JSON.

    Worth its ~5 s before a long arm: a missing or truncated checkpoint, a broken venv or a
    changed corgi.constants is then reported as a status instead of being discovered by a child
    that is already holding a leased card.
    """
    cmd = [child_python, "-P", os.path.abspath(__file__), "--probe", "--checkpoint",
           checkpoint]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "cmd": cmd}
    for l in reversed(p.stdout.splitlines()):
        if l.startswith("{"):
            try:
                return json.loads(l)
            except Exception:
                break
    return {"ok": False, "error": f"probe returned no JSON (rc {p.returncode}): "
                                  f"{(p.stderr or p.stdout)[-400:]}", "cmd": cmd}


def _unshadow():
    """THIS FILE IS NAMED corgi.py, and the child runs it as a script with the Corgi venv.

    Python then puts the script own directory first on sys.path, where this file would shadow the
    INSTALLED corgi package -- `from corgi.config import config_corgi` would resolve to this
    module and fail with "corgi is not a package".  The spawn passes -P (python >= 3.11) and this
    removes the directory as well, for a direct `python mrag_axis.arm/corgi.py` invocation.  The build
    tree above it is appended, not prepended, so s13 is importable without shadowing site-packages.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != here]
    up = os.path.dirname(here)
    if up not in sys.path:
        sys.path.append(up)


def main(argv=None):
    ap = argparse.ArgumentParser(description="the Corgi arm of the Section-13 axis validation")
    ap.add_argument("--worker", help="internal: run the GPU half from this job JSON")
    ap.add_argument("--probe", action="store_true",
                    help="load the checkpoint on the CPU and report, no GPU and no lease")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--edits", help="the edits JSON mrag_axis.prep wrote (the AlphaGenome arm's own)")
    ap.add_argument("--out", help="the JSONL to append to (resumable)")
    ap.add_argument("--run", help="the MRAG run directory the context and genome come from")
    ap.add_argument("--context", default=None, help="override the resolved expression context")
    ap.add_argument("--organism", default=None, help="override the run's recorded organism")
    ap.add_argument("--need-gib", type=float, default=None)
    ap.add_argument("--lock-dir", default=None)
    ap.add_argument("--limit-items", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=None)
    a = ap.parse_args(argv)
    _unshadow()
    if a.worker:
        return _worker(a.worker)
    if a.probe:
        print(json.dumps(_probe_child(a.checkpoint)))
        return 0
    if not (a.edits and a.out):
        ap.error("--edits and --out are required")
    rec = run(a.edits, a.out, a.run, explicit_context=a.context, explicit_genome=a.organism,
              need_gib=a.need_gib, lock_dir=a.lock_dir, limit_items=a.limit_items,
              checkpoint=a.checkpoint, timeout_s=a.timeout,
              log=lambda m: print(m, flush=True))
    print(json.dumps({k: rec[k] for k in ("arm", "status", "reason", "error", "verify")},
                     indent=1))
    return 0 if rec["status"] in (OK, NO_SIGNAL) else 1


if __name__ == "__main__":
    sys.exit(main())
