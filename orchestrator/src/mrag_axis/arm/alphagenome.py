"""The AlphaGenome arm of the Section-13 axis-validation node: the caller, its resume, its
retries and its status record.

The research caller is /BIO1/MRAG/corgi/ft/ag/agrun.py -- the only script that ever called
AlphaGenome for this section, so every AlphaGenome number in the 2026-10 tables came out of its
request and its window arithmetic.  This module is that caller, shape for shape: rebuild the
524-kb window from hg38, verify the prep's wt_md5 before using an item, apply the edits, one
client.predict_sequence(seq, requested_outputs=OUT, ontology_terms=ONT) per (label, tf, kind),
sum every track inside each named window of the item, append to a resumable JSONL with a flush
per row, and skip rows already present.

WHY agrun.py's SHAPE IS VENDORED instead of routed through the production client: the production
client (/BIO1/MRAG/Agent/libs/alphagenome_client) exposes only
predict_variant_with_retry(interval_, variant_, requested_outputs, tissue_curies, ...), which
cannot express a whole-sequence request at all, and its cache keys a request by hashing a small
JSON payload (cache.hash_request over chrom/pos/ref/alt), so it cannot key a 524288-base edited
sequence either.  Vendoring keeps the node's request identical to the one the published numbers
came from; test_arm_alphagenome.py pins that by importing agrun.py and comparing the OutputType
objects, the ontology terms and the sequence md5s.

WHAT THIS ADDS over the research runner, because a node has to report rather than merely finish:
  * a per-layer status record {attempted, done, missing, first_errors, md5_mismatches} so a
    partial run is reportable -- agrun.py logged skips to stdout and kept no tally;
  * retries with exponential backoff that RETURN the error instead of raising (agrun.predict
    raised RuntimeError after its 6 tries, which would abort the pipeline step), plus a
    consecutive-failure breaker so a dead API costs one backoff ladder, not one per job;
  * a status of ok | partial | no signal | failed carrying a reason, which is what the model-arm
    failure policy requires of this step: an AlphaGenome failure is a warning, the
    report says which arm is missing and why, and the run continues.

Nothing here raises for a missing input -- a missing edits file, a missing FASTA, an unimportable
SDK, an unobtainable client and a dead API are all a status and a reason.  The API key is read
through the production loader (alphagenome_client.config.load_api_key) at the single point where
the client is built, and is never returned, logged, stored in the status record or written to the
output.

Run with /BIO1/MRAG/Agent/libs/alphagenome_client/.venv/bin/python (the venv the published run
used: it has the SDK, and this module vendors agrun.py's FASTA reader because that venv has no
pyfaidx).
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, threading, time

import numpy as np

# The venv the published run used and the ONLY interpreter that has the SDK: the pipeline itself
# runs under /BIO1/MRAG/venvs/paper_qa, which does not, so run_layer_child() below spawns this
# file as a script under this python -- the same parent/child split mrag_axis.arm.corgi uses for
# torch.  Nothing in the parent imports the SDK.
AG_PY = "/BIO1/MRAG/Agent/libs/alphagenome_client/.venv/bin/python"

# hg38 as agrun.py read it (agrun.FASTA); the stored wt_md5 in every edits JSON is the md5 of a
# 524288-base window of THIS file, which is what makes the md5 check an integrity check.
HG38_FA = "/BIO1/MRAG/Agent/Reference/ATAC_hg38/hg38.fa"

# agrun.OUT, by name.  Names rather than SDK enums so this module imports without the AlphaGenome
# SDK present (an unimportable SDK must be a status, not an ImportError at node import time).
OUT_NAMES = ("DNASE", "ATAC", "CHIP_HISTONE", "RNA_SEQ", "CAGE")
# agrun.py --chip-tf appended CHIP_TF, which is how ft/prad/ag_A_prad_tf.jsonl was produced
# (prostate gland carries POLR2A / POLR2A-pS5 / CTCF); queue_ag_tf.sh ran both PRAD layers that way.
CHIP_TF = "CHIP_TF"

# agrun.MODALITIES: the response attribute each requested output arrives on, in request order.
# The order is load-bearing -- it fixes the key order of the row's "sums" dict and of the
# '.tracks.json' sidecar, and ft/prad/ag_A_prad_tf.jsonl's sidecar key order equals its row key
# order, so the published files were written in exactly this order.
MODALITY = {"DNASE": "dnase", "ATAC": "atac", "CHIP_HISTONE": "chip_histone",
            "RNA_SEQ": "rna_seq", "CAGE": "cage", "CHIP_TF": "chip_tf"}

# agrun.ONT.  Kept for the record and for the test that pins this module's request to the
# published one; a run derives its own terms through layer_request() below, not from this tuple.
ONT_PUBLISHED = ("EFO:0005726",      # LNCaP clone FGC: DNase, H3K4me3
                 "EFO:0002095",      # 22Rv1: H3K27ac
                 "EFO:0007752",      # VCaP: H3K27ac
                 "EFO:0002074",      # PC-3: ATAC, DNase, histones, RNA, CAGE
                 "UBERON:0002367")   # prostate gland: DNase, histones, RNA, CAGE

# The metadata columns agrun.py copied into the sidecar, and the only ones it copied.
SIDECAR_COLS = ("biosample_name", "ontology_curie", "histone_mark", "transcription_factor",
                "strand", "Assay title")

STATUSES = ("ok", "partial", "no signal", "failed")


# ---------------------------------------------------------------- sequence (agrun.py, verbatim)
class Fa:
    """Random access to an indexed FASTA without pyfaidx -- agrun.Fa, because the venv the
    published run used (libs/alphagenome_client/.venv) has the SDK but no pyfaidx."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.idx = {}
        for l in open(path + ".fai"):
            n, ln, off, lb, lw = l.split()[:5]
            self.idx[n] = (int(ln), int(off), int(lb), int(lw))

    def get(self, chrom, start, end):
        ln, off, lb, lw = self.idx[chrom]
        start, end = max(0, start), min(end, ln)
        self.f.seek(off + (start // lb) * lw + start % lb)
        n = end - start
        raw = self.f.read(n + n // lb + 2)
        return raw.replace(b"\n", b"").replace(b"\r", b"")[:n].decode().upper()

    def close(self):
        try:
            self.f.close()
        except Exception:  # noqa: BLE001 -- closing a FASTA is never a reason to fail a layer
            pass


def apply_edits(wt, edits):
    """agrun.apply_edits: every edit is a same-length substitution, so the sequence stays 524288
    bases and the control edit stays size-matched to the knockout."""
    s = list(wt)
    for pos, bases in edits:
        s[pos:pos + len(bases)] = bases
    return "".join(s)


def rebuild_wt(fa, item, window):
    """(wt, None) or (None, reason).  The two reasons are agrun.py's own two skips: a window that
    hg38 truncates at a contig end, and a sequence whose md5 differs from the one the prep step
    recorded -- which means the edit offsets in this item no longer point at the motifs they were
    scanned from, so the item is unusable rather than merely different."""
    wt = fa.get(item["chrom"], item["ws"], item["ws"] + window)
    if len(wt) != window:
        return None, "window truncated: %d of %d bases" % (len(wt), window)
    got = hashlib.md5(wt.encode()).hexdigest()
    if got != item["wt_md5"]:
        return None, "wt_md5 differs from prep: %s != %s" % (got, item["wt_md5"])
    return wt, None


def job_seqs(item, wt):
    """[(tf, kind, seq)] for one item, in agrun.py's order: the wild type first, then each
    factor's knockout and its size-matched control edit."""
    jobs = [("WT", "wt", wt)]
    for tf, e in item["tfs"].items():
        jobs.append((tf, "ko", apply_edits(wt, e["ko"])))
        jobs.append((tf, "ctl", apply_edits(wt, e["ctl"])))
    return jobs


def plan_jobs(items):
    """Every (label, tf, kind) the layer owes, in the order it would produce them."""
    out = []
    for it in items:
        out.append((it["label"], "WT", "wt"))
        for tf in it["tfs"]:
            out.append((it["label"], tf, "ko"))
            out.append((it["label"], tf, "ctl"))
    return out


# ---------------------------------------------------------------- read-out (agrun.py, verbatim)
def modalities_for(out_names):
    """The response attributes to read, in request order (agrun.MODALITIES)."""
    return tuple(MODALITY[n] for n in out_names if n in MODALITY)


def track_key(mod, md, j):
    """agrun.track_key.  '<modality>|<track name>|<strand>' -- the key every stored row and every
    '.tracks.json' sidecar is written with, so s13.stats can look a read-out up by name."""
    return "%s|%s|%s" % (mod, md["name"].iloc[j], md["strand"].iloc[j])


def window_sums(o, windows, ws, modalities):
    """{track: {window_name: summed signal}} for a prediction whose sequence starts at ws.

    agrun.window_sums, index arithmetic included: a window interval [a, b) in genomic coordinates
    covers bins i0 = (a - ws) // r up to i1 = ceil((b - ws) / r), clamped to the track, and the
    sum is over float32 values in the track's own resolution (1 base for DNase/ATAC/RNA/CAGE,
    128 bases for the ChIP tracks -- see the resolutions in results_A.jsonl.tracks.json).  Any
    change to this arithmetic changes every published AlphaGenome effect, so it is reproduced
    rather than rewritten.
    """
    res = {}
    for mod in modalities:
        td = getattr(o, mod, None)
        if td is None:
            continue
        v = np.asarray(td.values)
        r = int(td.resolution)
        md = td.metadata
        for j in range(v.shape[1]):
            d = {}
            for wn, ivs in windows.items():
                tot = 0.0
                for a, b in ivs:
                    i0 = max((a - ws) // r, 0)
                    i1 = min(-(-(b - ws) // r), v.shape[0])
                    if i1 > i0:
                        tot += float(v[i0:i1, j].sum())
                d[wn] = tot
            res[track_key(mod, md, j)] = d
    return res


def tracks_sidecar(o, modalities):
    """The '.tracks.json' payload agrun.py wrote beside the JSONL on its first prediction: the
    resolution and the descriptive metadata columns per track key.  s13.stats reads this sidecar
    to know which track is which read-out, so it is written in the same key order and with the
    same column set (SIDECAR_COLS)."""
    meta = {}
    for mod in modalities:
        td = getattr(o, mod, None)
        if td is None:
            continue
        md = td.metadata
        for j in range(len(md)):
            meta[track_key(mod, md, j)] = {
                "resolution": int(td.resolution),
                **{c: str(md[c].iloc[j]) for c in md.columns if c in SIDECAR_COLS}}
    return meta


def has_signal(o, modalities):
    """Does this prediction carry any usable value at all?

    'no signal' has to be distinguishable from 'failed' in the status, because AlphaGenome
    answers a request for a modality a biosample does not have by leaving it out of the response
    rather than erroring (alphagenome_client.config says so, and that is what MODALITY_COVERAGE
    exists for).  A response with no requested modality attached, or one whose every value is
    non-finite or zero, is an answer with nothing in it -- not an error, and not a result."""
    if o is None:
        return False
    for mod in modalities:
        td = getattr(o, mod, None)
        vals = getattr(td, "values", None) if td is not None else None
        if vals is None:
            continue
        try:
            v = np.asarray(vals)                 # no dtype copy: a res-1 track is 524288 x k
            if v.dtype.kind not in "fiu":
                continue
        except Exception:  # noqa: BLE001 -- an unreadable track is simply no signal
            continue
        if v.size == 0:
            continue
        fin = np.isfinite(v)
        if fin.any() and bool(np.any(v[fin] != 0)):
            return True
    return False


# ---------------------------------------------------------------- the request and the client
def resolve_outputs(out_names):
    """(agrun.OUT-equivalent list of dna_output.OutputType, None) or (None, reason).

    Resolved here, lazily, rather than at import: the SDK lives only in
    libs/alphagenome_client/.venv, and a node imported in a process without it must report a
    status instead of failing to import."""
    try:
        from alphagenome.models import dna_output
    except Exception as e:  # noqa: BLE001
        return None, "AlphaGenome SDK not importable (%s: %s)" % (type(e).__name__, e)
    out, bad = [], []
    for n in out_names:
        t = getattr(dna_output.OutputType, n, None)
        if t is None:
            bad.append(n)
        else:
            out.append(t)
    if bad:
        return None, "unknown AlphaGenome output type(s): %s" % ", ".join(bad)
    return out, None


def make_client(agent_root="/BIO1/MRAG/Agent"):
    """(client, None) or (None, reason).

    The ONLY place an API key is touched: the production loader
    alphagenome_client.config.load_api_key() (env ALPHAGENOME_API_KEY, else
    $ALPHAGENOME_KEY_FILE, else /BIO1/MRAG/secrets/.alphagenome_key).  The key is passed straight
    to dna_client.create and never bound to a name that reaches the status record, the log or the
    output file -- a missing key is the FileNotFoundError that loader raises, reported as a
    reason."""
    try:
        import sys
        p = "%s/libs/alphagenome_client" % agent_root
        if p not in sys.path:
            sys.path.insert(0, p)
        from alphagenome_client.config import load_api_key
        from alphagenome.models import dna_client
        return dna_client.create(api_key=load_api_key()), None
    except Exception as e:  # noqa: BLE001 -- no key and no SDK are both statuses, not crashes
        return None, "no AlphaGenome client (%s: %s)" % (type(e).__name__, e)


def layer_request(run_dir, agent_root="/BIO1/MRAG/Agent", chip_tf=False):
    """The per-run request, derived by stage 2a rather than re-derived here: the ontology terms
    from mrag_axis.stat.config.ag_biosamples(run_dir, need=('CHIP_HISTONE',)) -- every biosample of the
    run's lineage group that carries the histone read-outs the enhancer end needs -- and the
    output types from mrag_axis.stat.config.ag_outputs(), which is the modalities at least one chosen
    biosample covers.  For the published PRAD run both reduce to agrun.py's hardcoded ONT and OUT
    (test_arm_alphagenome.py checks exactly that).

    Returns {"terms", "out_names", "biosamples", "source", "reason"}; reason is None on success.
    Term ORDER is not part of the request: AlphaGenome returns its metadata sorted by curie
    (results_A.jsonl.tracks.json lists the DNase tracks EFO:0002074, EFO:0005726, UBERON:0002367
    while agrun.ONT asked in a different order), so the derived sorted list is the same request."""
    try:
        from mrag_axis.stat import config as C
        b = C.ag_biosamples(run_dir, need=("CHIP_HISTONE",), agent_root=agent_root)
        names = list(C.ag_outputs(run_dir, bios=b, agent_root=agent_root))
    except Exception as e:  # noqa: BLE001
        return {"terms": [], "out_names": [], "biosamples": {}, "source": None,
                "reason": "cannot derive the AlphaGenome request from %s (%s: %s)"
                          % (run_dir, type(e).__name__, e)}
    if chip_tf and CHIP_TF not in names:
        names.append(CHIP_TF)
    reason = None if b["terms"] else ("no biosample in this run's lineage carries the histone "
                                      "read-outs the enhancer end needs")
    return {"terms": list(b["terms"]), "out_names": names, "biosamples": b["biosamples"],
            "source": b["source"], "reason": reason}


def heal_tail(out_path, log=None):
    """Drop a half-written last line before appending to it; returns the bytes cut.

    mrag_axis.arm.corgi.heal_tail, ported (review 3, #7).  A child killed between write() and
    flush() -- by the node's watchdog, an OOM kill or a preempted job -- leaves a truncated JSON
    line.  done_keys() skips that line, so the next resume appended the following row straight
    onto it and reported 'ok' with nothing left to do, while mrag_axis.stat.stats.load(), which
    json-loads EVERY line, raised on the file from then on: re-running the output directory (the
    documented recovery) could never recover it.  Cutting the partial line makes that prediction
    simply get made again.
    """
    try:
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
    except OSError as e:
        if log:
            log("could not repair the last line of %s (%s: %s)" % (out_path, type(e).__name__, e))
        return 0
    if log:
        log("dropped a half-written last line of %s (%d bytes)" % (out_path, size - cut))
    return size - cut


# The watchdog deadline a layer's child gets when the caller names none (review 3, #3).  The
# Corgi arm has had one since review 2 (corgi.default_timeout_s); this arm had none, so a child
# hung on a socket with no timeout blocked the synchronous step -- and the run -- for ever, where
# a failed arm must only be a WARN.  Measured: the PRAD AlphaGenome arm took ~66 min for ~3,000
# predictions (~1.3 s each).  The per-prediction allowance is ~8x that and the base covers the
# retry ladder (5+10+20+40+80 s per job, three jobs before the breaker), so only a child that has
# really stopped is killed.
TIMEOUT_BASE_S = 1800.0
TIMEOUT_PER_PREDICTION_S = 10.0


def default_timeout_s(edits_path, out_path=None, limit_items=0):
    """The deadline for one layer: TIMEOUT_BASE_S + TIMEOUT_PER_PREDICTION_S per prediction the
    layer still owes (a resume owes only the rows not yet on disk).  Read at call time, so a
    caller (the suite) can shorten it."""
    try:
        items = json.load(open(edits_path))["items"]
    except Exception:  # noqa: BLE001 -- the child reports an unreadable edits file itself
        items = []
    if limit_items:
        items = items[:int(limit_items)]
    owed = set(plan_jobs(items)) - set(done_keys(out_path) if out_path else [])
    return TIMEOUT_BASE_S + TIMEOUT_PER_PREDICTION_S * max(1, len(owed))


def done_keys(path):
    """The (label, tf, kind) rows already in the output JSONL -- what makes the run resumable.
    A truncated last line (a run killed mid-write) is ignored rather than fatal, so the row is
    simply redone."""
    done = []
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for l in fh:
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                done.append((r["label"], r["tf"], r["kind"]))
            except Exception:  # noqa: BLE001
                continue
    return done


def predict_with_retry(client, seq, requested_outputs, ontology_terms, tries=6, base_wait=5.0,
                       sleep=time.sleep, log=None):
    """(prediction, None) or (None, 'ExcType: message') after `tries` attempts with waits
    base_wait * 2**k -- agrun.predict's own ladder (5, 10, 20, 40, 80 s over 6 tries).

    The one deliberate difference: agrun.predict raised RuntimeError when the ladder ran out,
    which inside a pipeline step aborts the step; here the error is returned so the layer can
    record it and carry on, per the model-arm failure policy."""
    last = "no attempt made"
    for k in range(max(1, int(tries))):
        try:
            return client.predict_sequence(seq, requested_outputs=requested_outputs,
                                           ontology_terms=ontology_terms), None
        except Exception as e:  # noqa: BLE001 -- returned, never raised
            last = "%s: %s" % (type(e).__name__, str(e)[:200])
            if k == int(tries) - 1:
                break
            wait = base_wait * 2 ** k
            if log:
                log("  API error (%s); retry in %gs" % (last, wait))
            sleep(wait)
    return None, last


# ---------------------------------------------------------------- the status record
def _blank(edits_path, out_path):
    """Every field exists from the start, so a layer that dies on its first input still emits a
    record the report can render instead of a hole."""
    return {"layer": os.path.basename(out_path), "out": out_path, "edits": edits_path,
            "status": "failed", "reason": "not run",
            "items": 0, "items_used": 0, "planned": 0, "already_done": 0,
            "attempted": 0, "calls": 0, "written": 0, "done": 0, "missing": 0,
            "no_signal": 0, "nonfinite_sums": 0, "tracks": 0, "n_errors": 0,
            "md5_mismatches": [], "first_errors": [], "missing_examples": [],
            "requested_outputs": [], "ontology_terms": [], "biosamples": {}, "healed_bytes": 0,
            "elapsed_s": 0.0}


def _classify(st):
    """ok | partial | no signal | failed, and the sentence the report prints.

    'no signal' is its own status because AlphaGenome drops a modality a biosample does not carry
    instead of erroring: a layer whose every answer came back empty is not a failure to reach the
    API, and the fix is a different biosample, not a retry."""
    if st["planned"] == 0:
        return "failed", "no items in %s" % os.path.basename(st["edits"])
    why = []
    if st["md5_mismatches"]:
        why.append("%d item(s) skipped on a wt_md5 mismatch (%s)"
                   % (len(st["md5_mismatches"]), st["md5_mismatches"][0]["label"]))
    if st["first_errors"]:
        why.append("%d API error(s), first: %s"
                   % (st["n_errors"], st["first_errors"][0]["error"]))
    if st["no_signal"]:
        why.append("%d prediction(s) carried no signal" % st["no_signal"])
    if st["done"] >= st["planned"]:
        return "ok", "%d/%d predictions on disk" % (st["done"], st["planned"])
    tail = ("; ".join(why) or "run stopped early") + \
           " -- reported, not raised"
    if st["done"] > 0:
        return "partial", "%d of %d predictions on disk: %s" % (st["done"], st["planned"], tail)
    if st["no_signal"] > 0 and not st["first_errors"]:
        return "no signal", "AlphaGenome answered %d request(s) with no usable track: %s" \
                            % (st["no_signal"], tail)
    return "failed", "no prediction completed: %s" % tail


class _Rows:
    """Append-with-flush-per-row, opened on the first row so a layer that never gets a prediction
    leaves no empty output behind to be mistaken for a finished run."""

    def __init__(self, path):
        self.path, self.fh = path, None

    def write(self, row):
        if self.fh is None:
            self.fh = open(self.path, "a")
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


# ---------------------------------------------------------------- the layer
def run_layer(edits_path, out_path, client=None, ontology_terms=None, out_names=None,
              chip_tf=False, requested_outputs=None, fasta=HG38_FA, limit_items=0,
              tries=6, base_wait=5.0, sleep=time.sleep, log=None,
              max_consecutive_failures=3, max_first_errors=5, max_examples=20,
              agent_root="/BIO1/MRAG/Agent", biosamples=None):
    """Run one AlphaGenome layer (an enhancer-end or target-end edits JSON) into a resumable
    JSONL and return its status record.  Never raises.

    `client` injects a client (what the suite does: a stub exposing predict_sequence); when None
    one is built through make_client(), which is the only path that touches the API key.
    `ontology_terms` and `out_names` come from layer_request(run_dir) -- the node derives them
    from the run, it does not hardcode agrun.py's prostate list.
    """
    t0 = time.time()
    st = _blank(edits_path, out_path)
    # Reported, not used: SECTION13_CONTRACT.md's axis_<domain>.json carries the read-out
    # biosamples next to the per-arm status so the section can name what it measured in.
    st["biosamples"] = dict(biosamples or {})

    try:
        ed = json.load(open(edits_path))
        items_all = ed["items"]
        window = int(ed["window"])
    except Exception as e:  # noqa: BLE001 -- a missing or malformed edits file is a status
        st["reason"] = "cannot read the edits JSON %s (%s: %s)" % (edits_path, type(e).__name__, e)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st

    items = items_all[:limit_items] if limit_items else items_all
    st["items"], st["planned"] = len(items_all), len(plan_jobs(items))
    names = tuple(out_names) if out_names else OUT_NAMES
    if chip_tf and CHIP_TF not in names:
        names = names + (CHIP_TF,)
    mods = modalities_for(names)
    st["requested_outputs"] = list(names)
    st["ontology_terms"] = list(ontology_terms or [])

    # BEFORE reading what is done: a half-written last line would otherwise be skipped here and
    # the next row appended onto it (heal_tail).
    st["healed_bytes"] = heal_tail(out_path, log)
    done = set(done_keys(out_path))
    planned = set(plan_jobs(items))
    st["already_done"] = len(done & planned)
    st["done"] = st["already_done"]
    if st["planned"] and not (planned - done):
        # Resume with nothing left to do must not build a client or read the key.
        st["missing"] = 0
        st["tracks"] = _n_tracks(out_path)
        st["status"], st["reason"] = _classify(st)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st
    if not st["planned"]:
        st["status"], st["reason"] = _classify(st)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st

    if not st["ontology_terms"]:
        st["reason"] = ("no AlphaGenome ontology terms for this run; layer_request() could not "
                        "derive a biosample group -- the arm is skipped and named in the report")
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st
    if requested_outputs is None:
        requested_outputs, why = resolve_outputs(names)
        if requested_outputs is None:
            st["reason"] = why
            st["elapsed_s"] = round(time.time() - t0, 3)
            return st
    if client is None:
        client, why = make_client(agent_root)
        if client is None:
            st["reason"] = why
            st["elapsed_s"] = round(time.time() - t0, 3)
            return st
    try:
        fa = Fa(fasta)
    except Exception as e:  # noqa: BLE001
        st["reason"] = "cannot open hg38 at %s (%s: %s)" % (fasta, type(e).__name__, e)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st

    rows = _Rows(out_path)
    side = out_path + ".tracks.json"
    meta_written = os.path.exists(side)
    run = 0            # consecutive jobs whose whole retry ladder failed
    used = 0           # items whose sequence passed the md5 gate and was predicted from
    stop = None
    try:
        for it in items:
            if stop:
                break
            wt, why = rebuild_wt(fa, it, window)
            if wt is None:
                # agrun.py logged this and moved on; the node has to be able to say which items
                # it dropped, because a dropped item silently shortens the pair table.
                st["md5_mismatches"].append({"label": it["label"], "reason": why})
                if log:
                    log("skip %s: %s" % (it["label"], why))
                continue
            used += 1
            for tf, kind, seq in job_seqs(it, wt):
                key = (it["label"], tf, kind)
                if key in done:
                    continue
                st["attempted"] += 1
                o, err = predict_with_retry(client, seq, requested_outputs, st["ontology_terms"],
                                            tries=tries, base_wait=base_wait, sleep=sleep,
                                            log=log)
                st["calls"] += 1
                if err is not None:
                    st["n_errors"] += 1
                    run += 1
                    if len(st["first_errors"]) < max_first_errors:
                        st["first_errors"].append({"label": it["label"], "tf": tf, "kind": kind,
                                                   "error": err})
                    if run >= max_consecutive_failures:
                        # A dead API costs one backoff ladder per breaker, not one per job: the
                        # quota and the step's wall clock are both finite.
                        stop = ("stopped after %d consecutive failed predictions" %
                                run)
                        break
                    continue
                run = 0
                if not has_signal(o, mods):
                    st["no_signal"] += 1
                    if log:
                        log("no signal for %s %s %s" % (it["label"], tf, kind))
                    continue
                if not meta_written:
                    try:
                        json.dump(tracks_sidecar(o, mods), open(side, "w"), indent=1)
                        meta_written = True
                    except Exception as e:  # noqa: BLE001 -- the sidecar is descriptive only
                        if log:
                            log("could not write %s (%s)" % (side, type(e).__name__))
                sums = window_sums(o, it["windows"], it["ws"], mods)
                st["nonfinite_sums"] += sum(1 for d in sums.values() for x in d.values()
                                            if x != x or x in (float("inf"), float("-inf")))
                rows.write({"label": it["label"], "tf": tf, "kind": kind, "sums": sums})
                st["written"] += 1
                done.add(key)
    except Exception as e:  # noqa: BLE001 -- the arm is a warning, never an exception
        stop = "unexpected %s: %s" % (type(e).__name__, str(e)[:200])
        if len(st["first_errors"]) < max_first_errors:
            st["first_errors"].append({"label": None, "tf": None, "kind": None, "error": stop})
            st["n_errors"] += 1
    finally:
        rows.close()
        fa.close()

    st["items_used"] = used
    st["done"] = len(done & planned)
    miss = sorted(planned - done)
    st["missing"] = len(miss)
    st["missing_examples"] = ["%s|%s|%s" % k for k in miss[:max_examples]]
    st["tracks"] = _n_tracks(out_path)
    st["status"], st["reason"] = _classify(st)
    if stop and st["status"] != "ok":
        st["reason"] = st["reason"] + " [" + stop + "]"
    st["elapsed_s"] = round(time.time() - t0, 3)
    if log:
        log("%s: %s -- %s" % (st["layer"], st["status"], st["reason"]))
    return st


def _n_tracks(out_path):
    """How many tracks the layer's sidecar describes; 0 when no prediction ever arrived."""
    try:
        return len(json.load(open(out_path + ".tracks.json")))
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------- the parent / child split
# WHY: run_layer() imports the AlphaGenome SDK (resolve_outputs) and builds the client, and the
# SDK lives ONLY in libs/alphagenome_client/.venv.  Production starts every run with
# /BIO1/MRAG/venvs/paper_qa/bin/mrag, where `import alphagenome` raises ModuleNotFoundError, so
# calling run_layer in-process makes the arm fail on every run.  The parent therefore writes a job
# JSON and runs THIS FILE as a script under the SDK's venv, exactly as mrag_axis.arm.corgi runs
# itself under the Corgi venv; the child returns the same per-layer status record, so the node's
# contract does not change.  The API key is never passed on the command line or in the job: the
# child reads it through alphagenome_client.config.load_api_key() (make_client), from the
# environment it inherits or from the key file.

_JOB_KEYS = ("edits", "out", "ontology_terms", "out_names", "biosamples", "chip_tf", "fasta",
             "limit_items", "tries", "base_wait", "agent_root", "max_consecutive_failures")


def child_env(base=None):
    """The child's environment: the parent's own, plus unbuffered output so a ~66-min layer's
    per-row log reaches the run log live.  ALPHAGENOME_API_KEY / ALPHAGENOME_KEY_FILE are
    inherited if the run has them, which is what the deleted ag_validation.py did too."""
    env = dict(os.environ if base is None else base)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _unshadow():
    """THIS FILE IS NAMED alphagenome.py, and the child runs it as a script.

    Python puts the script's own directory first on sys.path, where this file would shadow the
    INSTALLED alphagenome SDK -- `from alphagenome.models import dna_output` would resolve to this
    module and fail with "alphagenome is not a package", i.e. exactly the failure the child exists
    to avoid.  The spawn passes -P (python >= 3.11) and this removes the directory as well, for a
    direct `python mrag_axis/arm/alphagenome.py` invocation.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != here]


def _spawn(cmd, env, timeout_s, log, tag):
    """Run a child, stream its log into ours, and KILL it on the timeout even if it goes silent.

    The deadline is enforced by p.wait(timeout=...) on this thread while a reader thread pumps the
    output, because a check inside the read loop only fires when the child prints -- a child that
    hangs with no output (a CUDA init, a socket with no timeout) would never be killed, and this
    step is synchronous inside the run.
    """
    t0, tail = time.time(), []
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)

    def _pump():
        for line in p.stdout:
            line = line.rstrip("\n")
            if log:
                log("  [%s] %s" % (tag, line))
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
        killed = "killed after %gs (the child stopped before it finished)" % timeout_s
        try:
            p.wait(timeout=30)
        except Exception:  # noqa: BLE001 -- the record says it was killed either way
            pass
    th.join(timeout=10)
    return {"cmd": list(cmd), "returncode": p.returncode, "killed": killed,
            "elapsed_s": round(time.time() - t0, 2), "tail": "\n".join(tail)}


def run_layer_child(edits_path, out_path, child_python=AG_PY, timeout_s=None, log=None,
                    env=None, **kw):
    """run_layer() for ONE layer, executed in the SDK's own interpreter.  Never raises.

    Returns the child's status record (the same dict run_layer returns), or a 'failed' record
    carrying the child's exit code and the tail of its log when the child produced none.
    """
    t0 = time.time()
    st = _blank(edits_path, out_path)
    st["biosamples"] = dict(kw.get("biosamples") or {})
    job = {k: kw.get(k) for k in _JOB_KEYS if k in kw}
    job.update(edits=edits_path, out=out_path, result=out_path + ".layer.json")
    if not os.path.exists(child_python):
        st["reason"] = ("the AlphaGenome venv python is missing at %s, so the SDK cannot be "
                        "reached" % child_python)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st
    try:
        job_path = out_path + ".job.json"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(job_path, "w") as fh:
            json.dump(job, fh, indent=1)
        if os.path.exists(job["result"]):
            os.unlink(job["result"])          # never read a previous attempt's record
    except Exception as e:  # noqa: BLE001 -- an unwritable job file is a status
        st["reason"] = "cannot write the AlphaGenome job file: %s: %s" % (type(e).__name__, e)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st
    cmd = [child_python, "-P", os.path.abspath(__file__), "--worker", job_path]
    if not timeout_s:
        timeout_s = default_timeout_s(edits_path, out_path, kw.get("limit_items") or 0)
        if log:
            log("no timeout given; the AlphaGenome child gets the derived default %.0fs"
                % timeout_s)
    st["timeout_s"] = float(timeout_s)
    try:
        child = _spawn(cmd, child_env(env), timeout_s, log, "ag")
    except Exception as e:  # noqa: BLE001 -- a broken venv is a status, never an exception
        st["reason"] = "the AlphaGenome child could not be started: %s: %s" % (type(e).__name__, e)
        st["elapsed_s"] = round(time.time() - t0, 3)
        return st
    # A child the watchdog killed (or that died) mid-write leaves a half line, and the node scores
    # this file right after the arm returns: cut it here, not only on the next resume.
    healed = heal_tail(out_path, log)
    got, bad = None, ""
    try:
        if os.path.exists(job["result"]):
            with open(job["result"]) as fh:
                got = json.load(fh)
    except Exception as e:  # noqa: BLE001
        bad = "the AlphaGenome child's record is unreadable: %s: %s" % (type(e).__name__, e)
    if isinstance(got, dict) and got.get("status") in STATUSES:
        got["child"] = child
        got["timeout_s"] = st["timeout_s"]
        got["healed_bytes"] = int(got.get("healed_bytes") or 0) + healed
        if child.get("killed"):
            got["status"] = "partial" if got.get("done") else "failed"
            got["reason"] = (got.get("reason") or "") + " [" + child["killed"] + "]"
        got["elapsed_s"] = round(time.time() - t0, 3)
        return got
    st["child"] = child
    st["healed_bytes"] = healed
    # _blank() seeds reason with "not run", so this is assigned rather than defaulted: the child's
    # exit code, whether the watchdog killed it and the tail of its log are the only evidence a
    # layer with no record has.
    st["reason"] = "; ".join(x for x in (bad, (
        "the AlphaGenome child exited %s without a status record%s: %s"
        % (child.get("returncode"), (" (" + child["killed"] + ")") if child.get("killed") else "",
           (child.get("tail") or "")[-400:]))) if x)
    # The rows it did manage to append are still on disk, so the layer is partial, not lost.
    st["done"] = len(set(done_keys(out_path)))
    st["tracks"] = _n_tracks(out_path)
    st["status"] = "partial" if st["done"] else "failed"
    st["elapsed_s"] = round(time.time() - t0, 3)
    return st


def probe(child_python=AG_PY, out_names=OUT_NAMES, timeout_s=120):
    """The parent-side preflight: resolve the SDK's output types IN THE CHILD, no client and no
    request.  Worth its ~2 s before a ~66-min layer: a venv without the SDK is then a status
    before the first API call rather than after it."""
    if not os.path.exists(child_python):
        return {"ok": False, "python": child_python, "reason": "no such interpreter"}
    cmd = [child_python, "-P", os.path.abspath(__file__), "--probe", ",".join(out_names)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "python": child_python,
                "reason": "%s: %s" % (type(e).__name__, str(e)[:200])}


def _worker(job_path):
    """The child half: run one layer and write its status record where the parent will read it."""
    with open(job_path) as fh:
        job = json.load(fh)
    kw = {k: job[k] for k in _JOB_KEYS if k in job and k not in ("edits", "out")}
    st = run_layer(job["edits"], job["out"], log=lambda m: print(m, flush=True), **kw)
    with open(job["result"], "w") as fh:
        json.dump(st, fh, indent=1, default=str)
    print("layer %s: %s" % (st.get("status"), st.get("reason")), flush=True)
    return 0 if st.get("status") in ("ok", "partial", "no signal") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="the AlphaGenome arm of the Section-13 axis test")
    ap.add_argument("--worker", help="internal: run one layer from this job JSON")
    ap.add_argument("--probe", nargs="?", const=",".join(OUT_NAMES),
                    help="resolve the SDK's output types and report; no client, no request")
    a = ap.parse_args(argv)
    _unshadow()
    if a.worker:
        return _worker(a.worker)
    if a.probe:
        names = tuple(n for n in a.probe.split(",") if n)
        out, why = resolve_outputs(names)
        print(json.dumps({"ok": out is not None, "python": sys.executable,
                          "requested": list(names), "resolved": len(out or []),
                          "reason": why}))
        return 0 if out is not None else 1
    ap.error("--worker or --probe is required")


if __name__ == "__main__":
    sys.exit(main())
