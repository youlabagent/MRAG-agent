"""The Section-13 axis-validation node: SE-TF-target edges tested in sequence space.

Replaces the single-base-substitution step and its single-base scatter, per the Section-13
design contract (SECTION13_CONTRACT.md).  What it reports is the pair-level
enrichment of the agreed definition, measured by two models on the SAME edits:

  1. mrag_axis.stat.config derives every per-run constant -- the ROSE table and its signal column,
     the SE table, the HOMER peak table, the TRN factor list and its JASPAR filter, the
     lineage-foreign null, the AlphaGenome biosample group and output types, the Corgi channel map
     and window geometry.  Nothing here names a run or a run id.
  2. mrag_axis.prep writes the enhancer-end and target-end edit items (one RNG stream each).
  3. mrag_axis.arm replays them through AlphaGenome and Corgi, CONCURRENTLY: the AlphaGenome arm
     is network-bound and the Corgi arm GPU-bound, measured at ~66 min and ~86 min on the
     published PRAD run, so running them serially costs ~2.5 h against ~90 min in parallel.
  4. mrag_axis.stat scores the stored per-pair results with definition="unified" -- the definition
     the report prints -- and this node writes axis_validation/axis_<domain>.json in the shape the
     contract specifies, plus a per-arm status record.

NOTHING HERE FAILS THE RUN (PI decision 3; the model-arm failure policy, for AlphaGenome
and for the GPU).  Each arm's failure is outputs["axis_<arm>"]["status"] plus a
manifest WARN gate and a stated reason, and a step that cannot run at all -- no TRN, no ROSE
table, or a non-human run (gated in the runner, since Corgi is hg38-only and AlphaGenome's
catalogue is human) -- records a skip the report can state instead of raising.

TWO INPUTS ARE NOT DERIVABLE from a run and are therefore node params, because the published
numbers and the derivable default disagree and both have to stay reachable (PI decision 5):

  trn_order     the factor iteration order, which fixes which draws of the single RNG stream go to
                which (item, factor).  The OC2 research order is seko2.TRN_TFS, this run's TRN
                typed alphabetically ("sorted"); PRAD used the run's own se_cyjs TF-node order
                ("network"), the one order a node can derive, and so the default here.
  target_order  the network GENE order, which fixes where in the stream each network item's draws
                sit.  "sorted" reproduces the published OC2 edits, "network" the PRAD ones.
                Default "network", same reason.

Both are recorded in the node's outputs AND in the axis JSON: the published OC2 numbers need the
research values and PRAD needs the derived ones, so a reader of either must be able to see which
was used.  mrag_axis.prep.edits_enhancer.TRN_ORDER and edits_target.TARGET_ORDER state the
measurement behind each.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ._base import Node

# The report prints AlphaGenome first, so the arms are recorded in that order.
ARMS = ("alphagenome", "corgi")
# The two layers of edits, in the order mrag_axis.prep builds them.  Each arm walks both.
LAYERS = ("enhancer", "target")
# The SE -> closest-gene end (2026-10-20).  An SE -> SE_gene edge of the TRN names no factor --
# it says this super-enhancer is the closest one to that gene -- so it is measured without one:
# the super-enhancer is scrambled (its gene's promoter and exons spared) and that gene is read.
# Two layers, which are the two control families: "se_direct" carries the knock-out and 21
# controls placed anywhere in the window, "se_near" 22 more placed as close to the gene as the
# window allows.  They are kept OUT of rec["arms"][arm]["layers"]: arm_status() takes the worst
# layer of that record, and this end is DESCRIPTIVE like the triples, so a failure here must
# never degrade an arm whose enhancer and target ends measured fine.
SE_LAYERS = ("se_direct", "se_near")

# The arm statuses that are not a failure of the arm (mrag_axis.arm.corgi's vocabulary; the
# AlphaGenome arm's STATUSES is a subset).  Anything outside "ok" earns a manifest WARN gate.
# "not built" is the one status that is the RUN's, not the arm's: the layer has no edits because
# mrag_axis.prep could not build them (no promoter peak for any target gene, say), so neither arm
# was asked to measure it.  It ranks above "ok" and arm_status() ignores it while another layer
# exists -- an arm that measured everything it was given is "ok", and the missing end is stated
# once, as the run's target_withheld (review 3, #15).
NOT_BUILT = "not built"
ARM_STATUSES = ("ok", "partial", "no signal", "skipped", "no context", "unsupported (hg38 only)",
                "failed", NOT_BUILT)
# Worst first: an arm's status is the worst of its two layers, because a half-measured arm cannot
# be reported as a measurement.
_SEVERITY = {s: i for i, s in enumerate(("failed", "unsupported (hg38 only)", "no context",
                                         "skipped", "no signal", "partial", "ok", NOT_BUILT))}
# The model code mrag_axis.stat.stats uses for each arm, and back.
MODEL_OF = {"alphagenome": "ag", "corgi": "corgi"}
ARM_OF = {v: k for k, v in MODEL_OF.items()}
# The values run.toml can declare for the two orders and the target read-out cells (config.py
# validates them at load time; the node also accepts an explicit list for the orders).
ORDERS = ("network", "sorted")
TARGET_CELLS = ("", "auto", "tissue", "cell_lines")
# A Corgi gate that is a property of the RUN, not of the layer: re-running the second layer would
# ask for a card, a context or an hg38 genome that the first layer already established is absent.
_CORGI_RUN_GATES = ("unsupported (hg38 only)", "no context", "skipped")


def _slug(text, fallback="run"):
    """A file-name-safe domain label for axis_<domain>.json."""
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(text or "").strip().lower())
    s = "_".join(p for p in s.split("_") if p)
    return s or fallback


def _n_rows(path):
    """Rows in a per-pair result JSONL, 0 when there is no readable file."""
    try:
        with open(path) as fh:
            return sum(1 for l in fh if l.strip())
    except OSError:
        return 0


def _clear_stand_in(path):
    """Remove a '{}' stand-in sidecar (and the empty JSONL beside it) left at an arm's own result
    path, BEFORE that arm runs.  Returns what it removed.

    An earlier version of this node wrote such a pair for a layer that produced nothing, so that
    the statistic -- which loads both arms' files -- would not take the arm that worked down with
    it.  That broke every re-run of the same output directory, which is the documented recovery:
    the real AlphaGenome arm writes its sidecar only `if not os.path.exists(side)`, so it appended
    every row and left the '{}' in place (the AlphaGenome columns then never appear), and the real
    Corgi arm refuses a sidecar that describes a different track set than its own ('[]'), so it
    failed for good.  mrag_axis.stat.stats.load() now treats a MISSING file as an empty result,
    which is what "this model has no column" means, so no stand-in is needed at all.
    """
    gone = []
    side = path + ".tracks.json"
    try:
        if os.path.exists(side) and os.path.getsize(side) <= 4:
            with open(side) as fh:
                empty = json.load(fh) == {}
            if empty:
                os.unlink(side)
                gone.append(os.path.basename(side))
                if os.path.exists(path) and os.path.getsize(path) == 0:
                    os.unlink(path)
                    gone.append(os.path.basename(path))
    except Exception:                      # noqa: BLE001 -- a sidecar we cannot read is left
        return gone                        # alone; the arm itself then reports what it finds
    return gone


def _md5(path):
    """The md5 of a file, "" when it cannot be read."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


# The result files one (arm, layer) owns, all keyed to the same edits: the rows, the track
# sidecar, the arms' own status records and the child's job/result files.
_RESULT_SUFFIXES = ("", ".tracks.json", ".status.json", ".worker.json", ".job.json",
                    ".layer.json")


def _fingerprint(result_path, edits_path, log=None):
    """Keep the stored per-pair results only while they belong to THESE edits.

    Both arms resume on (label, tf, kind) alone, and every execution rewrites edits_<layer>.json
    from the run's own TRN -- so after an upstream fix and a re-run in the same output directory
    the arms would skip every existing key and the section would score the PREVIOUS edits'
    predictions as this run's.  The edits md5 is therefore stored beside each arm's results and
    compared here; on a mismatch the stale files are moved aside (never deleted: they are hours of
    API quota and GPU time) and the arm starts fresh.  Returns (the files moved aside, the md5).
    """
    want = _md5(edits_path)
    side = result_path + ".edits.json"
    have = ""
    try:
        with open(side) as fh:
            have = str(json.load(fh).get("edits_md5") or "")
    except Exception:                      # noqa: BLE001 -- no fingerprint yet, or unreadable
        have = ""
    moved = []
    if want and os.path.exists(result_path) and not have:
        # Results with NO fingerprint: written by the version of this node that did not store one,
        # in this same output directory and from the same derivation.  There is nothing to compare
        # them against, so they are adopted (and fingerprinted from here on) rather than thrown
        # away -- discarding them would cost a re-run of the whole arm on a guess.
        if log:
            log("%s carries no edits fingerprint (written before this node stored one); adopting "
                "it for the current edits" % os.path.basename(result_path))
    elif want and os.path.exists(result_path) and have != want:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for suf in _RESULT_SUFFIXES:
            p = result_path + suf
            if os.path.exists(p):
                try:
                    os.replace(p, "%s.stale-%s" % (p, stamp))
                    moved.append(os.path.basename(p))
                except OSError:
                    pass
        if log:
            log("the edits changed since %s was written (md5 %s -> %s); %d stale file(s) moved "
                "aside, the arm starts fresh"
                % (os.path.basename(result_path), have[:8] or "none", want[:8], len(moved)))
    if want:
        try:
            with open(side, "w") as fh:
                json.dump({"edits": edits_path, "edits_md5": want,
                           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
        except OSError:
            pass
    return moved, want


def _sidecar_keys(paths):
    """Every track key the given layers' sidecars describe -- what the run actually measured."""
    keys = set()
    for p in paths:
        try:
            with open(p + ".tracks.json") as fh:
                d = json.load(fh)
        except Exception:                  # noqa: BLE001 -- a missing sidecar is "no tracks"
            continue
        if isinstance(d, dict):
            keys |= set(d)
    return keys


def _readout_cells(S, ag_paths, declared=""):
    """(the target-end read-out cells, a label, the reason) for this run's own biosamples.

    DECLARED wins (node param target_cells: "tissue" = TGT_PRAD, "cell_lines" = TGT_OC2;
    review 3, #5 -- not yet a run.toml key: the runner does not pass it): every prostate run requests the same five biosamples, prostate tissue
    included, so the derived choice below always lands on the tissue set and the published
    pc/nec/sclc read-out (LNCaP / 22Rv1) is reachable only by declaring it.

    mrag_axis.stat.stats keys its read-outs on ontology curies, so WHICH cell set applies is a
    property of the biosamples the run derived, not of this node: TGT_PRAD reads prostate TISSUE
    (UBERON:0002367) and TGT_OC2 the LNCaP / 22Rv1 cell lines, and the published PRAD numbers were
    measured with the first while pc/nec/sclc used the second.  Pick the set whose AlphaGenome
    tracks this run's own sidecars actually carry, preferring the tissue set; the Corgi keys
    (corgi|rna_total, corgi|h3k4me3, corgi|dnase) are identical in both.
    """
    keys = _sidecar_keys(ag_paths)
    cand = (("prostate tissue (UBERON:0002367)", S.TGT_PRAD),
            ("LNCaP / 22Rv1 cell lines", S.TGT_OC2))
    d = str(declared or "").strip().lower()
    if d in ("tissue", "cell_lines"):
        label, cells = cand[0] if d == "tissue" else cand[1]
        want = [asub for (asub, _aw), (_cs, _cw) in cells.values()]
        if keys and all(any(sub in k for k in keys) for sub in want):
            return cells, label, None
        return cells, label, ("declared (target_cells = %s), but this run's "
                              "AlphaGenome tracks (%d measured) do not carry that set's "
                              "read-outs, so its AlphaGenome target end has no column to score; "
                              "the Corgi read-out keys are the same in both cell sets"
                              % (d, len(keys)))
    if not keys:
        return S.TGT_PRAD, cand[0][0], ("no AlphaGenome track metadata for this run; the Corgi "
                                        "read-out keys are the same in both cell sets")
    for label, cells in cand:
        want = [asub for (asub, _aw), (_cs, _cw) in cells.values()]
        if all(any(sub in k for k in keys) for sub in want):
            return cells, label, None
    return S.TGT_PRAD, cand[0][0], ("no read-out cell set matches this run's AlphaGenome tracks "
                                    "(%d track(s) measured); the target end cannot be scored"
                                    % len(keys))


def _read_biosamples(requested, ag_paths, scored=None):
    """{biosample: curie} the statistic READ OUT, out of the ones the AlphaGenome arm requested.

    The request is the whole lineage group (PROSTATE_CELL_LINES for a prostate run), so naming it
    in the section said the test was "read out on PC-3, LNCaP_FGC, 22Rv1, VCaP, prostate_gland"
    when the statistic read prostate tissue and used 22Rv1/VCaP only for the AR gate -- and PC-3
    was not read at all.  Matching the curies against the layer's '.tracks.json' sidecar alone
    did not fix that: the arm writes the sidecar from every track the response carries for the
    whole requested group, so every requested biosample matched (review 3, #4).

    `scored` is the list of track-key substrings the statistic actually scores this run (the
    enhancer read-outs; the target read-out cells when that end is scored; the 22Rv1 / VCaP
    H3K27ac tracks when the declared AR gate is on).  A biosample counts as read when its curie is
    named by one of those substrings AND a track matching that substring is on disk in a layer's
    sidecar.  Default: the enhancer read-outs only.
    """
    keys = _sidecar_keys(ag_paths)
    if not keys:
        return {}
    if scored is None:
        from mrag_axis.stat import stats as S
        scored = (S.TRACKS["h3k27ac_prostate"], S.TRACKS["dnase_prostate"])
    present = [s for s in scored if any(s in k for k in keys)]
    curies = {c for s in present for c in re.findall(r"[A-Za-z]+:[0-9]+", s)}
    return {b: c for b, c in (requested or {}).items() if c and str(c) in curies}


def _models_with_columns(frame, reads):
    """Which of ('ag', 'corgi') the frame carries every read-out column for.

    mrag_axis.stat.stats.beats() indexes the column by name, so asking it for an arm that wrote no
    rows is a KeyError, not an empty result.  This is the check that turns a failed arm into "that
    model is not reported, and here is why" instead of an exception.
    """
    have = set(getattr(frame, "columns", []))
    return [m for m in ("ag", "corgi")
            if all(("%s_%s" % (m, rd)) in have for rd in reads)]


def _holm(by):
    """Holm over the headline test of every model set this run REPORTS.

    The family is the set of headline tests actually printed, which is what prices the
    multiplicity in; the target end and the triples are descriptive and are not in it.
    report.holm itself cannot be called here: it walks all three model sets of all four research
    domains, and a run that lost an arm reports fewer -- so the family is this run's own
    headlines.  The headline is the AR-gated row where the run declares one, the enhancer row
    otherwise, which is report.headline_key's rule.

    STEP-DOWN MONOTONE: the adjusted values are the running maximum over the sorted family, which
    is what Holm's step-down procedure rejects on.  Without it the column is not Holm -- a less
    significant test can print a SMALLER adjusted p (0.02 / 0.026 / 0.03 came out as 0.06 / 0.052
    / 0.03, so the 0.03 read as surviving although Holm stops at the first test and rejects
    nothing), and two tied raw p values got different adjusted ones.  mrag_axis.stat.report.holm
    keeps the arithmetic it was measured with, because its 12 published 2026-10 values are
    reproducible numbers (PI decision 6) and 6 of them -- the pc/nec/sclc ties, which are ONE
    measurement reported in three domains -- would move; this family is the run's own.
    """
    # A headline row that could not be tested (p null, testable false: see _pairs) is not a test,
    # so it is not in the family and gets no Holm p.
    heads = sorted((v[_headline(v)]["p"], m) for m, v in by.items()
                   if v[_headline(v)].get("p") is not None)
    adj = itertools.accumulate((min(1.0, (len(heads) - i) * q)
                                for i, (q, _m) in enumerate(heads)), max)
    return {m: v for v, (_q, m) in zip(adj, heads)}


def _headline(res_one_model):
    """report.headline_key's rule, stated on this node's own key names."""
    return "pair_enhancer_ar" if "pair_enhancer_ar" in res_one_model else "pair_enhancer"


HEADLINES = ("both", "alphagenome", "corgi")


def choose_headline(models, declared=""):
    """(the model set that leads Section 13, why) -- SECTION13_CONTRACT.md, PI decision 2026-10-18.

    DECLARED wins whenever that set was scored (run.toml axis_headline; PRAD declares
    "alphagenome").  Otherwise the consensus leads when both arms ran and BOTH single-model
    headline tests survive Holm; otherwise the one single model whose headline survives.  It is
    never "the smallest p": that rule picks AlphaGenome for OC2 and gets the science backwards
    (the consensus leads there on its false-positive rate).  It decides the default tab and the
    emphasised row and nothing else -- every scored set is printed.
    """
    d = str(declared or "").strip().lower()
    if d and d in models:
        return d, "declared by the run: axis_headline = %s" % d
    pre = ""
    if d:
        pre = ("the declared %s set was not scored (%s); "
               % (d, "unknown value" if d not in HEADLINES else "its arm produced no result"))

    def survives(m):
        h = (models.get(m) or {}).get(_headline(models.get(m) or {})) or {}
        return h.get("holm") is not None and h["holm"] < 0.05

    surv = [m for m in ("alphagenome", "corgi") if m in models and survives(m)]
    if "both" in models and len(surv) == 2:
        return "both", pre + ("default: both arms ran and both single-model headline tests "
                              "survive Holm, so the consensus leads")
    if len(surv) == 1:
        return surv[0], pre + ("default: only the %s headline test survives Holm" % surv[0])
    if surv:
        # Both single models survive, but the consensus set itself was not scored (one arm wrote
        # no rows at the other end, say), so there is no consensus row to lead with.
        lead = next(m for m in HEADLINES if m in surv)
        return lead, pre + ("default: both single-model headline tests survive Holm but the "
                            "consensus set was not scored, so the %s set leads" % lead)
    first = next((m for m in HEADLINES if m in models), "")
    return first, pre + ("default: no single-model headline test survives Holm; the %s set is "
                         "shown first" % first if first else "no model set was scored")


def _model_cfg(scfg, mods, own_rows):
    """`scfg` with the arm a SINGLE-model set does not use pointed at a path that does not exist.

    Both arms' columns live in one frame, and the frame builders drop a row that lacks any of
    them, so a half-finished arm silently shrinks the OTHER arm's denominators: with the
    AlphaGenome layer half on disk, Corgi's enhancer end went from 18/18 against 48 foreign to
    9/9 against 24, with a different p.  A single-model set is therefore framed on ITS OWN rows
    whenever an arm is incomplete, and only the consensus is intersected across both arms -- which
    is what "both models" means.  With both arms complete the two are the same rows, so this is
    off by default and the stored 2026-10 re-issue (stage 2d) is unaffected.

    PER END AND PER ARM (review 3, #2): `own_rows` is either True (every other arm's file, both
    ends -- the old rule) or {"enhancer": [model codes incomplete there], "target": [...]}, and a
    single-model set drops the OTHER arm's file only at an end where THAT arm is incomplete.  A
    complete layer covers every row, so keeping it shrinks nothing -- and it is what keeps the
    declared AR row on the Corgi set: the AR gate is an AlphaGenome read-out (ag_ar22 / ag_arvc)
    taken from the AlphaGenome ENHANCER layer, so an AlphaGenome target layer that ran out of
    quota no longer strips Corgi's "PRAD + AR" row (130/981) and moves its Holm headline.
    """
    if not own_rows or set(mods) >= {"ag", "corgi"}:
        return scfg
    own = ({"enhancer": ("ag", "corgi"), "target": ("ag", "corgi")} if own_rows is True
           else own_rows)
    absent = os.path.join(os.path.dirname(str(scfg.get("enh_edits") or ".")),
                          "__no_such_arm__.jsonl")
    c, changed = dict(scfg), False
    for m, enh_keys, tgt_keys in (("ag", ("enh_ag", "enh_ag_pub"), ("ag",)),
                                  ("corgi", ("enh_cg",), ("cg",))):
        if m in mods:
            continue
        for end, keys in (("enhancer", enh_keys), ("target", tgt_keys)):
            if m in (own.get(end) or ()):
                c.update({k: absent for k in keys})
                changed = True
    return c if changed else scfg


def _model_frames(scfg, mods, e, b, own_rows, ends):
    """(the cfg, the enhancer frame, the target frame) for one model set."""
    from mrag_axis.stat import stats as S
    cfg_m = _model_cfg(scfg, mods, own_rows)
    if cfg_m is scfg and e is not None:
        return cfg_m, e, b
    e_m, _drop = S.enh_frame(None, definition="unified", excl="rule", cfg=cfg_m)
    b_m = (None if ends == "enhancer"
           else S.tgt_frame(None, definition="unified", src="all", cfg=cfg_m))
    return cfg_m, e_m, b_m


def score_models(scfg, e, b, usable, min_rd=1, why_missing=None, own_rows=False,
                 ends="both", ar_gate=False, usable_target=None, target_why=None, notes=None):
    """({model set: result}, {withheld set: reason}) on the frames e (enhancer) and b (target).

    The scoring half of AxisValidationNode._score, as a function of the stored per-pair results
    only, so the four 2026-10 reports (stage 2d) are re-issued through the SAME code a new run
    uses.  Holm is this run's own family of headline tests (see _holm).

    own_rows   score a single-model set on its own complete rows (see _model_cfg): True, or
               {end: [model codes incomplete at that end]}; the node passes the second, and it
               changes nothing when both arms are complete.
    ends       "both", or "enhancer" when the target end has no edits: the target end is
               DESCRIPTIVE (PI decision 1), so losing it must not discard a measured enhancer end,
               which is the end the evidence rests on.  The triples need both ends and are then
               not reported.
    ar_gate    the run DECLARED the AR cross-condition requirement (run.toml axis_ar_gate); it is
               never a default, so an undeclared run gets no AR row even on prostate read-outs.
    usable     the model codes with every ENHANCER read-out column; a set needing another is
               withheld.  usable_target: those with every TARGET read-out column (default: the
               same list, the stage-2d call).  PER MODEL SET (review 3, #0): a set whose model
               lacks the target end -- one arm's target layer refused its GPU lease, say -- keeps
               its measured enhancer end and has only its target row and triples withheld, as
               ends="enhancer" does for the whole run; target_why(model code) says why.
    notes      a dict this fills with what the axis JSON states beside the rows:
               targets_withheld {set: reason}, ar_withheld {set: reason} (the run declares the AR
               row but this set could not carry it), headline_rows {set: its Holm row} and
               se_tf_edges {set: _se_tf_edges(...)}, the network SE -> TF edges the figure draws.
    """
    from mrag_axis.stat import stats as S
    why_missing = why_missing or (lambda m: "no read-out column on disk")
    target_why = target_why or (lambda m: "no target-end read-out column on disk")
    usable_target = list(usable) if usable_target is None else list(usable_target)
    notes = {} if notes is None else notes
    for k in ("targets_withheld", "ar_withheld", "headline_rows", "se_tf_edges"):
        notes.setdefault(k, {})
    ar_on = bool(ar_gate) or scfg.get("branch") == "prad"
    models, withheld = {}, {}
    for name, mods in S.MODELS.items():
        miss = [m for m in mods if m not in usable]
        if miss:
            withheld[name] = "; ".join("%s: %s" % (m, why_missing(m)) for m in miss)
            continue
        tmiss = [m for m in mods if m not in usable_target] if ends == "both" else []
        ends_m = "enhancer" if (ends == "enhancer" or tmiss) else "both"
        cfg_m, e_m, b_m = _model_frames(scfg, mods, e, b, own_rows, ends_m)
        if ends_m == "enhancer":
            # enhancer_only adds the AR row exactly when _axis_unified would: branch "prad" (the
            # declared gate, or a published PRAD cfg) AND the AlphaGenome AR columns in the frame.
            r = S.enhancer_only(cfg_m, e_m, models=mods, min_rd=min_rd, ar=ar_on)
            m = {"pair_enhancer": _pairs(r["enhancer"])}
            if "enhancer_plus_AR" in r:
                m["pair_enhancer_ar"] = _pairs(r["enhancer_plus_AR"])
            notes["se_tf_edges"][name] = _se_tf_edges(
                S, cfg_m, e_m, S.rowsets(e_m, "label", "label", S.ENH_RD, mods, min_rd=min_rd),
                "enhancer_plus_AR" in r, m)
            if tmiss:
                notes["targets_withheld"][name] = (
                    "the target end of this set was not scored: %s. Its enhancer end, which is "
                    "the end the evidence rests on, is reported in full; the SE-TF-target "
                    "triples need both ends and are not drawn."
                    % "; ".join("%s: %s" % (ARM_OF.get(x, x), target_why(x)) for x in tmiss))
            models[name] = m
            continue
        gene = dict(zip(zip(b_m.label, b_m.tf), b_m.gene))
        r = S.axis(None, definition="unified", src="all", min_rd=min_rd, models=mods,
                   excl="rule", cfg=cfg_m)
        r.pop("dropped_pairs", None)
        E = S.rowsets(e_m, "label", "label", S.ENH_RD, mods, min_rd=min_rd)
        B = {(gene[k], k[1]) for k in
             S.rowsets(b_m, "gene", "label", S.TGT_RD, mods, min_rd=min_rd)}
        # triple_stats' second return value is the axes themselves, which _axis_unified drops
        # and the contract's triples{axes_list} needs for the graph.
        tri, axes = S.triple_stats(cfg_m, e_m, b_m, E, B)
        keys = ("axes", "n_triples", "false", "n_foreign", "SE", "TF", "target")
        triples = {k: tri[k] for k in keys}
        triples["axes_list"] = sorted([list(x) for x in axes])
        # Descriptive only: no Fisher test on triples (PI decision 1), and this records that
        # the axes drawn are the same ones the statistic counted.
        triples["agrees_with_statistic"] = all(r["graph_triples"][k] == triples[k]
                                               for k in keys)
        m = {"pair_enhancer": _pairs(r["enhancer"]), "pair_target": _pairs(r["target"]),
             "triples": triples}
        if "enhancer_plus_AR" in r:
            m["pair_enhancer_ar"] = _pairs(r["enhancer_plus_AR"])
        notes["se_tf_edges"][name] = _se_tf_edges(S, cfg_m, e_m, E, "enhancer_plus_AR" in r, m)
        models[name] = m
    if scfg.get("branch") == "prad":
        for name, m in models.items():
            if "pair_enhancer_ar" not in m:
                # Stated, not silent: this set's Holm headline is then its plain enhancer row
                # while the sets that carry the AR row lead with that one.
                notes["ar_withheld"][name] = (
                    "the AR cross-condition row is an AlphaGenome read-out (22Rv1 and VCaP "
                    "H3K27ac from the AlphaGenome enhancer layer), and this set's frame carries "
                    "no such column -- it is scored on its own rows because the AlphaGenome "
                    "enhancer layer is incomplete or missing -- so it has no AR row and its Holm "
                    "headline is its plain enhancer row (pair_enhancer)")
    h = _holm(models)
    for name, m in models.items():
        notes["headline_rows"][name] = _headline(m)
        for k in ("pair_enhancer", "pair_enhancer_ar", "pair_target"):
            if k in m:
                # Holm covers only the headline test each model set reports; the target end
                # and the triples are descriptive and are not in the family (contract).
                m[k]["holm"] = h.get(name) if k == _headline(m) else None
    return models, withheld


def _nan(v):
    return isinstance(v, float) and v != v


def _se_tf_edges(S, cfg, e, E, ar_row, m):
    """The network's own SE -> TF edges for the Section-13 figure (PI decision 2026-10-22).

    The figure colours an edge only where a group-level test supports the step: the enhancer end
    separates network factors from lineage-foreign ones, the target end does not, so no
    TF -> target edge is recorded here.  Of the SE -> TF edges the run's network draws (cfg["net"],
    the list _grid builds the triples from; `n_graph` of them, `n_no_motif` naming a factor the
    statistic does not score, stats.norm -> None), `tested` are the ones with a row in the
    enhancer frame and `passed[row]` the ones in the pass set that pair row counted: E for
    pair_enhancer, E and the AR gate for pair_enhancer_ar, the same expressions
    stats.enhancer_only and _axis_unified use.  `agrees_with_statistic` recounts every network
    pair of the frame in each pass set against the row's own `passed`, so an edge list that does
    not belong to the printed row is never drawn as if it did.

    `subset[row]` tests the drawn edges themselves: `passed` of `tested` against the row's own
    lineage-foreign pairs (one-sided Fisher, stats.fisher).  The row pools every region -- the
    super-enhancers and the sampled ordinary enhancers -- so a significant row does not by itself
    say the edges the graph draws pass above the lineage-foreign rate (PRAD's consensus AR row:
    3 of 145 drawn edges against 3.8% of foreign pairs).  Descriptive and outside the Holm family;
    the report does not gate its figure on it (user decision 2026-10-22).

    Descriptive, like the triples: anything that goes wrong here is this set's stated `error`,
    never a failed scoring of the run.
    """
    try:
        passes = {"pair_enhancer": E}
        if ar_row:
            passes["pair_enhancer_ar"] = (E & S.beats(e, "label", "label", "ag_ar22")
                                          & S.beats(e, "label", "label", "ag_arvc"))
        with open(cfg["net"]) as fh:
            net = json.load(fh)
        drawn = {(str(s_), str(t)) for s_, t in (net.get("se_tf") or [])}
        graph = {(s_, S.norm(t)) for s_, t in drawn if S.norm(t) is not None}
        no_motif = sorted({t for _s, t in drawn if S.norm(t) is None})
        cls = dict(zip(zip(e.label, e.tf), e.tf_class))
        network = [k for k in zip(e.label, e.tf) if cls[k] != "foreign"]
        tested = sorted({k for k in network if (str(k[0]), S.norm(k[1])) in graph})
        agrees = all(len([k for k in network if k in P]) == (m.get(row) or {}).get("passed")
                     for row, P in passes.items())
        subset = {}
        for row, P in passes.items():
            r = m.get(row) or {}
            a, fp, nf = len([k for k in tested if k in P]), r.get("fp") or 0, r.get("n_frn") or 0
            o, p = (S.fisher(a, len(tested), fp, nf) if tested and nf else (float("nan"), float("nan")))
            subset[row] = {"passed": a, "tested": len(tested), "fp": fp, "n_frn": nf,
                           "odds_ratio": None if (o is None or _nan(o)) else o,
                           "odds_ratio_infinite": o is None, "p": None if _nan(p) else p}
        return {"n_graph": len(drawn),
                "n_no_motif": len([1 for _s, t in drawn if S.norm(t) is None]), "no_motif_tfs": no_motif,
                "tested": [list(k) for k in tested],
                "passed": {row: [list(k) for k in tested if k in P] for row, P in passes.items()},
                "subset": subset, "agrees_with_statistic": agrees}
    except Exception as ex:                # noqa: BLE001 -- descriptive figure data
        return {"error": "%s: %s" % (type(ex).__name__, ex)}


def _pairs(d):
    """One axis-JSON pair row: the contract's {passed, n, fp, n_frn, odds_ratio, p} (+ holm),
    plus two flags that say what a null odds ratio means (review 3, #6):

      testable             False when nothing could be tested: fisher_exact returns NaN for a
                           table with an empty column -- no pair passed in either group, every
                           pair passed in both, or a group with no pairs at all.  odds_ratio AND
                           p are then null (never p = 1, which reads as a calibrated negative) and
                           untestable_reason says which case it is.  Such a row is not in the
                           Holm family.
      odds_ratio_infinite  True when the test ran and the odds ratio is INFINITE (no
                           lineage-foreign pair passed while network pairs did): odds_ratio is
                           null because JSON has no infinity, and p is the real p.

    A NaN is never written as a bare token (no strict JSON reader accepts it).
    """
    out = {k: d[k] for k in ("passed", "n", "fp", "n_frn")}
    orr, p = d.get("odds_ratio"), d.get("p")
    a, n, c, nf = (d.get(k) or 0 for k in ("passed", "n", "fp", "n_frn"))
    why = None
    if not n or not nf:
        why = ("the table is empty: %d network and %d lineage-foreign pair(s), so there is no "
               "comparison to test" % (n, nf))
    elif a + c == 0:
        why = ("no pair passed in either group (0/%d vs 0/%d), so the odds ratio is undefined and "
               "nothing could be tested" % (n, nf))
    elif a == n and c == nf:
        why = ("every pair passed in both groups (%d/%d vs %d/%d), so the odds ratio is undefined "
               "and nothing could be tested" % (a, n, c, nf))
    elif _nan(orr) or p is None or _nan(p):
        why = "the odds ratio is undefined for this table, so nothing could be tested"
    if why:
        out.update(odds_ratio=None, p=None, testable=False, odds_ratio_infinite=False,
                   untestable_reason=why)
    else:
        out.update(odds_ratio=orr, p=p, testable=True, odds_ratio_infinite=orr is None)
    return out


def _rule(fn, *a, **kw):
    """(value, reason) for a mrag_axis.stat.config rule that returns that pair, with an exception
    turned into the reason: a rule that cannot be satisfied is a reported fallback, never an
    exception (config.py's own contract, because this step has no on/off switch)."""
    try:
        return fn(*a, **kw)
    except Exception as e:                 # noqa: BLE001 -- reported, not raised
        return None, "%s: %s" % (type(e).__name__, e)


def _value(fn, *a, **kw):
    """The same, for a config rule that returns a single value."""
    try:
        return fn(*a, **kw), None
    except Exception as e:                 # noqa: BLE001 -- reported, not raised
        return None, "%s: %s" % (type(e).__name__, e)


def readout_curies():
    """The ontology curies the ENHANCER-end read-out tracks name.

    mrag_axis.stat.stats reads the enhancer end off two fixed track substrings (prostate-tissue
    H3K27ac and DNase), so a run whose AlphaGenome request does not include those biosamples has
    no AlphaGenome enhancer column to score -- whatever it predicts.  Read out of stats' own
    TRACKS rather than written down here, so the two cannot drift apart.
    """
    from mrag_axis.stat import stats as S
    subs = (S.TRACKS["h3k27ac_prostate"], S.TRACKS["dnase_prostate"])
    return sorted({c for s in subs for c in re.findall(r"[A-Za-z]+:[0-9]+", s)})


def ag_readout_check(terms):
    """(can this run's AlphaGenome request be scored at the enhancer end, the reason it cannot).

    Decided BEFORE the arm runs.  The read-out track names are prostate, so a breast or pancreas
    run used to spend its whole quota (~66 min of API calls) and then have both the AlphaGenome
    and the consensus set withheld as "no read-out column on disk", which named the symptom and
    not the cause.  Deriving the read-out tracks from an arbitrary biosample is not a rename --
    the statistic's TGT_PRAD / TGT_OC2 cell sets and ENH_RD substrings are the measurement -- so
    the honest action is to skip the arm with the reason and spend nothing.
    """
    need = readout_curies()
    have = [str(t) for t in (terms or [])]
    miss = [c for c in need if c not in have]
    if not miss:
        return True, None
    return False, ("this run's AlphaGenome biosamples (%s) do not carry the enhancer-end read-out "
                   "tracks the statistic reads (%s: H3K27ac and DNase), so no AlphaGenome column "
                   "could be scored from them; the arm is skipped BEFORE any API call instead of "
                   "spending the quota on tracks this section cannot use"
                   % (", ".join(have) or "none", ", ".join(miss)))


def run_alphagenome(edits, outs, run_dir, agent_root, log=None, child_python=None,
                    timeout_s=None, in_process=False, layers=LAYERS, **kw):
    """The AlphaGenome arm: both layers through mrag_axis.arm.alphagenome, IN A CHILD PROCESS.

    The request is DERIVED, not hardcoded: layer_request(run_dir) is
    mrag_axis.stat.config.ag_biosamples(need=('CHIP_HISTONE',)) + ag_outputs(), i.e. every
    biosample of this run's lineage group that carries the histone read-outs the enhancer end
    needs.  Never raises -- the arm is a WARN gate (the model-arm failure policy).

    THE CHILD IS THE POINT.  run_layer() needs the AlphaGenome SDK, which lives only in
    libs/alphagenome_client/.venv; production starts every run with
    /BIO1/MRAG/venvs/paper_qa/bin/mrag, where `import alphagenome` raises ModuleNotFoundError.
    Run in-process the arm therefore failed on EVERY run, with the consensus and the AlphaGenome
    set withheld each time.  run_layer_child() runs the layer under that venv's python and returns
    the same per-layer status record, the way the Corgi arm runs itself under the Corgi venv and
    the deleted ag_validation.py ran its own runner.  `in_process=True` is the suite's seam for a
    stubbed client (no SDK needed); the node never sets it.
    """
    from mrag_axis.arm import alphagenome as AG
    rec = {"arm": "alphagenome", "layers": {}}
    req = AG.layer_request(run_dir, agent_root=agent_root)
    rec["request"] = {k: req.get(k) for k in ("terms", "out_names", "source", "reason")}
    rec["biosamples"] = req.get("biosamples") or {}
    ok, why = ag_readout_check(req.get("terms"))
    rec["readout_check"] = {"ok": bool(ok), "need": readout_curies(),
                            "terms": list(req.get("terms") or []), "reason": why}
    if not ok:
        if log:
            log("skipped before any API call: " + why)
        rec["layers"] = {l: {"status": "skipped", "reason": why} for l in layers}
        return rec
    runner = AG.run_layer if in_process else AG.run_layer_child
    if not in_process:
        kw = dict(kw, child_python=(child_python or AG.AG_PY), timeout_s=timeout_s)
    for layer in layers:
        if not edits.get(layer):
            rec["layers"][layer] = {"status": NOT_BUILT, "reason":
                                    "no %s-end edits were built" % layer}
            continue
        try:
            rec["layers"][layer] = runner(
                edits[layer], outs[layer], ontology_terms=req.get("terms") or None,
                out_names=req.get("out_names") or None, biosamples=rec["biosamples"],
                agent_root=agent_root, log=log, **kw)
        except Exception as e:             # noqa: BLE001 -- neither runner raises; a bad
            rec["layers"][layer] = {       # argument or a broken deployment would
                "status": "failed",
                "reason": "the AlphaGenome arm could not be run: %s: %s" % (type(e).__name__, e)}
    return rec


class _Borrowed:
    """The held lease as mrag_axis.arm.corgi.run's `with lease as gpu:` sees it: entering yields
    the real lease (a gpu_lease.Lease, or the falsy refusal with its .reason), leaving does NOT
    release it -- _SharedLease.release() does, once, after both layers."""

    def __init__(self, lease):
        self._lease = lease

    def __enter__(self):
        return self._lease

    def __exit__(self, *_a):
        return False

    def __bool__(self):
        return bool(self._lease)

    def __getattr__(self, k):
        return getattr(self._lease, k)


class _SharedLease:
    """ONE GPU lease for both Corgi layers (review 3, #0).

    corgi.run() leases a card per call, so the arm asked twice per run: the enhancer layer got a
    card and wrote every row, and when the target layer asked again the target scorer could
    already hold it -- the target layer came back "skipped", which used to withhold the Corgi set
    and the consensus (OC2's declared headline) although their enhancer end was measured in full.
    The lease is taken lazily, at the point the first layer's own gates (edits, organism,
    expression context, runtime preflight) have passed, and held until both layers are done.
    """

    def __init__(self, factory):
        self._factory, self.lease, self.requests = factory, None, 0

    def get(self):
        if self.lease is None:
            self.requests += 1
            self.lease = self._factory()
        return _Borrowed(self.lease)

    def release(self):
        lease, self.lease = self.lease, None
        if lease is not None:
            try:
                lease.close()
            except Exception:              # noqa: BLE001 -- gpu_lease.Lease.close never raises
                pass


def run_corgi(edits, outs, run_dir, agent_root, log=None, layers=LAYERS, **kw):
    """The Corgi arm: both layers through mrag_axis.arm.corgi.run, under ONE GPU lease that this
    function holds across both (see _SharedLease).

    Imported through mrag_axis.arm rather than directly: the two arms run in different
    interpreters by design (the SDK lives in libs/alphagenome_client/.venv, torch in
    libs/corgi_client/.venv), so the package records a corgi import failure instead of raising
    and this reports it as the arm's status.  Never raises (the model-arm failure policy).
    """
    from mrag_axis import arm as ARM
    rec = {"arm": "corgi", "layers": {}}
    if ARM.corgi is None:
        why = "mrag_axis.arm.corgi is not importable: %s" % ARM.corgi_import_error
        rec["layers"] = {l: {"status": "failed", "reason": why} for l in layers}
        return rec
    factory = kw.pop("lease_fn", None) or (
        lambda: ARM.corgi.lease_gpu(need_gib=kw.get("need_gib"), lock_dir=kw.get("lock_dir"),
                                    log=log))
    held = _SharedLease(factory)
    gate = None
    try:
        for layer in layers:
            if not edits.get(layer):
                rec["layers"][layer] = {"status": NOT_BUILT, "reason":
                                        "no %s-end edits were built" % layer}
                continue
            if gate:
                # No card, no expression context and a non-hg38 genome are properties of the
                # RUN, so the second layer would ask for the same missing thing
                # (the model-arm failure policy).
                rec["layers"][layer] = dict(gate, layer=layer)
                continue
            try:
                r = ARM.corgi.run(edits[layer], outs[layer], run_dir=run_dir,
                                  agent_root=agent_root, log=log, lease_fn=held.get, **kw)
            except Exception as e:         # noqa: BLE001 -- run() does not raise; a bad argument
                r = {"status": "failed",   # or a broken deployment would
                     "reason": "the Corgi arm could not be run: %s: %s" % (type(e).__name__, e)}
            rec["layers"][layer] = r
            if r.get("status") in _CORGI_RUN_GATES:
                gate = {"status": r["status"], "reason": r.get("reason") or "",
                        "inherited_from": layer}
    finally:
        held.release()
    rec["lease"] = {"requests": held.requests, "shared_across_layers": True}
    return rec


class _SeNearPrep:
    """mrag_axis.prep.edits_se_direct.build_near behind the (build, published_form) pair
    _build_edits calls, so the near control family is written and reported like any other layer."""

    @staticmethod
    def build(run, **kw):
        from mrag_axis.prep import edits_se_direct as SD
        return SD.build_near(run, **kw)

    @staticmethod
    def published_form(res):
        from mrag_axis.prep import edits_se_direct as SD
        return SD.published_form(res)


def arm_status(rec):
    """(status, reason) for one arm: the worst of its layers, named."""
    layers = rec.get("layers") or {}
    if not layers:
        return "failed", "the arm produced no layer record"
    worst = min(layers, key=lambda l: _SEVERITY.get(layers[l].get("status"), -1))
    st = layers[worst].get("status") or "failed"
    if st == NOT_BUILT:
        # Every layer is one the run could not build: the arm was asked to measure nothing.
        return "skipped", "; ".join("%s end: %s" % (l, layers[l].get("reason") or st)
                                    for l in LAYERS if l in layers)
    if st not in ARM_STATUSES:
        # A status this node does not know cannot be ranked, and treating it as "ok" would print
        # an unmeasured arm as a measurement.
        return "failed", "%s end reported a status this node does not know: %r" % (worst, st)
    if st == "ok":
        return "ok", ""
    return st, "%s end: %s" % (worst, layers[worst].get("reason") or st)


def _incomplete_layers(arms):
    """{end: [model codes]} -- the arms whose layer at that end is not complete.

    "ok" is complete and "not built" is the run's (no edits to measure, for either arm); any other
    layer status, or an arm with no layer record at all, is incomplete at that end.
    """
    out = {l: [] for l in LAYERS}
    for arm in ARMS:
        layers = (arms.get(arm) or {}).get("layers") or {}
        for l in LAYERS:
            st = (layers.get(l) or {}).get("status")
            if st not in ("ok", NOT_BUILT):
                out[l].append(MODEL_OF[arm])
    return out


def _arm_effects(rec):
    """{arm: what that arm's state does to the OTHER numbers}, for an arm that is not "ok".

    The WARN gate printed one fixed sentence -- "the other arm is scored on its own complete
    rows, so its pair-level numbers do not depend on this arm" -- which was false whenever the
    Corgi set lost its declared AR row with the AlphaGenome arm (review 3, #2).  Built from what
    was actually scored, so the gate says only what is true of this run.
    """
    models = rec.get("models") or {}
    own = rec.get("own_rows_by_end") or {}
    tw = rec.get("targets_withheld") or {}
    arw = rec.get("ar_withheld") or {}
    withheld = rec.get("models_withheld") or {}
    out = {}
    for arm in ARMS:
        if (rec["arms"].get(arm) or {}).get("status") == "ok":
            continue
        other = "corgi" if arm == "alphagenome" else "alphagenome"
        bits = []
        if other in models:
            ends = [l for l in LAYERS if arm in (own.get(l) or [])]
            if ends:
                bits.append("the %s set is scored on its own complete rows at the %s end%s, so "
                            "those pair-level rows do not depend on this arm"
                            % (other, " and ".join(ends), "s" if len(ends) > 1 else ""))
            if other in arw:
                bits.append("the %s set has NO AR-gated row: %s" % (other, arw[other]))
            if other in tw:
                bits.append("the %s set reports its enhancer end only: %s" % (other, tw[other]))
        else:
            bits.append("the %s set is not reported either (%s)"
                        % (other, withheld.get(other) or "not scored"))
        if "both" in models:
            bits.append("the two-model consensus is scored on the rows both arms share%s"
                        % (" (enhancer end only)" if "both" in tw else ""))
        else:
            bits.append("the two-model consensus needs both arms and is withheld")
        out[arm] = "; ".join(bits)
    return out


class AxisValidationNode(Node):
    """Inputs:
        run_output:   this run's output directory.  mrag_axis.stat.config.run_output resolves a run
                      id, a run directory or an output directory, and every other per-run constant
                      is derived from it -- nothing is passed in that the run already records.
        se_cyjs_json: se_trn/se_cyjs.json, the TRN whose edges are validated.  Named as an input,
                      although config.net_edges() finds it itself, so that a rebuilt TRN makes
                      this step dirty (Node.input_hash).
        se_top_tsv:   the ROSE SE table, for the same reason.
       Params:
        domain:         the label axis_<domain>.json carries; derived from the run's own
                        cancer_type / tissue / cell_line when empty.
        trn_order:      "network" (default, derivable) | "sorted" (the OC2 research order,
                        seko2.TRN_TFS) | an explicit factor list.
        target_order:   "network" (default; reproduces PRAD) | "sorted" (reproduces OC2) | an
                        explicit gene list.  Node params only: the runner passes "network" for both and
                        no run.toml key sets them yet (review 3, #5, a README limit).
        target_cells:   "" / "auto" (default: derived from the sidecars, see _readout_cells) |
                        "tissue" (TGT_PRAD) | "cell_lines" (TGT_OC2, the published pc/nec/sclc
                        read-out) -- node param only; the runner does not pass it.
        headline:       "" | both | alphagenome | corgi -- the model set that leads Section 13
                        (run.toml axis_headline); see choose_headline().
        ar_gate:        False (default) | True -- the AR cross-condition extra requirement
                        (run.toml axis_ar_gate).  DECLARED, never a default: every
                        prostate-lineage AlphaGenome request carries the 22Rv1 / VCaP tracks, so
                        an undeclared run must not get the AR row as its Holm headline.
        literature_domain: this run's own literature domain list, joined -- the section prints it,
                        and only the runner resolves it (it is NOT run.toml's cancer_type text).
        ag_python / ag_timeout_s: the interpreter the AlphaGenome layers run in (default the SDK's
                        own venv) and the watchdog deadline for that child.
        min_rd:         read-outs a row must pass, 1 (the declared set prices the multiplicity in).
        n_se / n_ctl / score_frac / max_motifs / max_peaks_per_gene: the prep parameters, at the
                        research values; changing one changes the edits and so the measurement.
        need_gib / lock_dir / corgi_timeout_s: handed to the Corgi arm's GPU lease.
        ag_tries / ag_base_wait: handed to the AlphaGenome arm's retry ladder.
       Outputs:
        axis_json:        axis_<domain>.json, the shape SECTION13_CONTRACT.md specifies, plus the
                          keys the report reads directly: window (bp), literature_domain,
                          biosamples (the ones the AlphaGenome arm actually read), default_model
                          and headline_reason.  A run whose target end could not be built carries
                          target_withheld (the reason) and no pair_target / triples.
        axis_alphagenome / axis_corgi: the per-arm status record {status, reason, error, rows}
        axis_prep:        the two edit builds' status
        axis_step:        the step's own {status, reason, domain, trn_order, target_order, ...}
        n_axes / n_models
    """
    name = "axis_validation"

    # ---------------------------------------------------------------- derivation
    def _log(self, msg):
        print("[%s] %s" % (self.name, msg), flush=True)

    def _arm_log(self, arm):
        return lambda m: print("[%s/%s] %s" % (self.name, arm, m), flush=True)

    def _derive(self, C, run_out, agent_root):
        """Every per-run constant this node uses, each with the reason for the ones it cannot get.

        Recorded in the axis JSON because the section has to name what it measured on, and because
        a withheld number must come with the rule that could not be satisfied (config.fallback()).
        """
        d = {"run_output": run_out, "agent_root": agent_root, "missing": {}, "notes": {}}
        toml = C.run_toml(run_out)
        d["domain_text"] = (toml.get("cancer_type") or toml.get("tissue")
                            or toml.get("cell_line") or "")
        for key, got in (
                ("rose_all", _rule(C.rose_all, run_out)),
                ("se10", _rule(C.se10, run_out)),
                ("peak_table", _rule(C.peak_table, run_out)),
                ("net_edges", _rule(C.net_edges, run_out)),
                ("trn_tf_names", _rule(C.trn_tf_names, run_out)),
                ("lineage", _rule(C.lineage, run_out, agent_root)),
        ):
            val, why = got
            if val is None:
                d["missing"][key] = why or "not derivable"
                continue
            if why:
                d["notes"][key] = why
            if key == "net_edges":
                d["n_tfs"], d["n_targets"] = len(val["tfs"]), len(val["targets"])
                d["n_se_tf"], d["n_tf_target"] = len(val["se_tf"]), len(val["tf_target"])
                d["net_edges"] = val
            elif key == "trn_tf_names":
                keep, drop = C.split_pwm(val)
                d["trn_tfs"], d["no_pwm"] = keep, drop
                neg, nwhy = _rule(C.neg_tfs, run_out, keep, None, agent_root)
                d["foreign_tfs"], d["foreign_source"] = (neg or []), nwhy
                if not neg:
                    d["notes"]["neg_tfs"] = "%s -- %s" % (nwhy, C.fallback("neg_tfs"))
            elif key == "rose_all":
                col, cwhy = _rule(C.rose_signal_col, val)
                d["rose_all"], d["rose_signal_col"] = val, col
                if cwhy:
                    d["notes"]["rose_signal_col"] = cwhy
            else:
                d[key] = val if not isinstance(val, dict) else val
        d["ag_request"], why = _value(C.ag_biosamples, run_out, ("CHIP_HISTONE",), agent_root)
        if why:
            d["missing"]["ag_terms"] = why
        d["corgi"], why = _value(C.corgi_channels)
        if why:
            d["missing"]["corgi_constants"] = why
        d["expr_context"], why = _value(C.expr_context, run_out, agent_root)
        if why:
            d["missing"]["expr_context"] = why
        elif not (d["expr_context"] or {}).get("rna_available"):
            d["notes"]["expr_context"] = C.fallback("expr_context")
        return d

    # ---------------------------------------------------------------- the step
    def run(self) -> None:
        """NOTHING HERE MAY RAISE (PI decision 3).  Node.execute turns a raise into a step status
        'fail' with no gates and no axis JSON, and Manifest.finalize then fails the WHOLE run --
        so every path is guarded here and becomes a recorded status, a stated reason and a manifest
        WARN gate instead.  The step is default-on with no switch, which is exactly why a bad
        config rule, a missing reference file or an unreadable result must not end the run."""
        t0 = time.time()
        out_dir = Path(self.output_dir)
        st = {"rec": None, "path": None}
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            return self._run(out_dir, st, t0)
        except Exception as e:             # noqa: BLE001 -- recorded, never raised
            why = ("the axis step raised and was recorded instead of failing the run: %s: %s"
                   % (type(e).__name__, e))
            rec = st["rec"] or {
                "domain": _slug(self.params.get("domain")), "definition": "unified",
                "status": "failed", "reason": "", "config": {"missing": {}, "notes": {}},
                "prep": {}, "arms": {}, "models": {}, "models_withheld": {}, "dropped_pairs": [],
                "read_outs": {}, "biosamples": {}, "foreign_tfs": [], "default_model": "",
                "headline_reason": "", "headline_declared": str(self.params.get("headline") or ""),
                "trn_order": self.params.get("trn_order", "network"),
                "target_order": self.params.get("target_order", "network"),
                "min_rd": int(self.params.get("min_rd", 1)), "window": None,
                "literature_domain": str(self.params.get("literature_domain") or ""),
                "ar_gate_declared": bool(self.params.get("ar_gate"))}
            rec["status"] = "failed"
            rec["reason"] = "; ".join(x for x in (rec.get("reason"), why) if x)
            rec["error"] = why
            for arm in ARMS:
                a = rec.setdefault("arms", {}).setdefault(
                    arm, {"arm": arm, "layers": {}, "rows": {l: 0 for l in LAYERS}})
                if not a.get("status"):
                    a["status"], a["reason"], a["error"] = "failed", why, why
            self._log("FAILED, recorded not raised: " + why)
            path = st["path"] or (out_dir / ("axis_%s.json" % (rec.get("domain") or "run")))
            try:
                return self._write(rec, path, t0)
            except Exception as e2:        # noqa: BLE001 -- the emit is guarded too
                self.outputs = self._bare_outputs(
                    "%s | and the axis JSON could not be written: %s: %s"
                    % (why, type(e2).__name__, e2))
                return None

    def _bare_outputs(self, why):
        """The outputs of a step that could not even write its record: the shape the runner and
        validate() read, so the section still states that Section 13 is missing and why."""
        return {
            "axis_json": "",
            "axis_alphagenome": {"status": "failed", "reason": why, "error": why,
                                 "rows": {l: 0 for l in LAYERS}},
            "axis_corgi": {"status": "failed", "reason": why, "error": why,
                           "rows": {l: 0 for l in LAYERS}},
            "axis_prep": {l: {"status": "failed", "reason": why, "n_items": 0} for l in LAYERS},
            "axis_step": {"status": "failed", "reason": why,
                          "domain": _slug(self.params.get("domain")), "trn_order": "",
                          "target_order": "", "headline": "", "headline_reason": "",
                          "models": [], "withheld": {}, "elapsed_s": 0.0},
            "n_axes": 0, "n_models": 0,
        }

    def _run(self, out_dir, st, t0):
        from mrag_axis.stat import config as C

        agent_root = str(self.params.get("agent_root") or C.AGENT_ROOT)
        given = str(self.inputs.get("run_output") or out_dir.parent)
        run_out = C.run_output(given, agent_root)
        if not os.path.isdir(run_out) and os.path.isdir(given):
            # run_output() resolves a run id and a run directory, and appends "/output" to a
            # directory that is neither named that nor carries an output_dir in its run.toml.
            # When that path does not exist, the directory the runner handed us IS the output.
            run_out = given
        cfg = self._derive(C, run_out, agent_root)
        trn_order = self.params.get("trn_order") or "network"
        target_order = self.params.get("target_order") or "network"
        target_cells = str(self.params.get("target_cells") or "").strip().lower()
        bad = [("%s=%r" % (k, v)) for k, v in (("trn_order", trn_order),
                                               ("target_order", target_order))
               if isinstance(v, str) and v not in ORDERS]
        if target_cells not in TARGET_CELLS:
            bad.append("target_cells=%r" % target_cells)
        min_rd = int(self.params.get("min_rd", 1))
        domain = _slug(self.params.get("domain") or cfg["domain_text"])
        axis_path = out_dir / ("axis_%s.json" % domain)
        rec = {"domain": domain, "run_output": run_out, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               # definition="unified" is what the report prints; definition="published" stays in
               # mrag_axis.stat only to regenerate the 2026-10 artefacts (PI decision 5).
               "definition": "unified", "min_rd": min_rd,
               "trn_order": trn_order, "target_order": target_order,
               "target_cells_declared": target_cells if target_cells != "auto" else "",
               "headline_declared": str(self.params.get("headline") or ""),
               "default_model": "", "headline_reason": "",
               # The window every edit is cut from and the run's OWN literature domain, both of
               # which the section's approved intro prints ("elsewhere in the same 524-kb window",
               # the literature domain row).  The domain here is the slug of the first literature
               # domain; literature_domain is the run's whole list, which is NOT the run.toml
               # cancer_type text (the runner passes it, since only the runner resolves it).
               "window": int((cfg.get("corgi") or {}).get("WINDOW") or 0) or None,
               "literature_domain": str(self.params.get("literature_domain") or ""),
               "ar_gate_declared": bool(self.params.get("ar_gate")),
               "status": "ok", "reason": "", "config": cfg, "prep": {}, "arms": {},
               "models": {}, "models_withheld": {}, "dropped_pairs": [], "read_outs": {},
               "biosamples": {}, "foreign_tfs": list(cfg.get("foreign_tfs") or [])}
        st["rec"], st["path"] = rec, axis_path
        if bad:
            # config.py rejects these at load time; a direct caller with a typo gets a stated
            # skip, never a silently different measurement.
            return self._skip(rec, axis_path, "unknown axis parameter value(s): %s (the orders "
                              "take %s or an explicit list, target_cells %s)"
                              % (", ".join(bad), "|".join(ORDERS),
                                 "|".join(x for x in TARGET_CELLS if x)), t0)

        # The TRN that gets validated is the one THIS run built.  config.net_edges finds a
        # se_cyjs.json by itself, and on a re-run where the se_trn step failed that is the EARLIER
        # run's file sitting in the same output directory -- ~90 minutes spent validating a
        # network this run did not build, reported as this run's axes.
        trn_ok, trn_why = self._trn_input(C, run_out)
        if not trn_ok:
            return self._skip(rec, axis_path, trn_why, t0)

        # Two inputs are the step itself: without a TRN there are no axes to validate, and
        # without the ROSE table there is no enhancer end -- which is where the evidence is.
        blocked = [k for k in ("net_edges", "rose_all") if k in cfg["missing"]]
        if blocked:
            return self._finish_skipped(rec, axis_path, blocked, C, t0)

        rec["prep"] = prep = self._build_edits(out_dir, run_out, agent_root, trn_order,
                                               target_order)
        rec["foreign_tfs"] = sorted({t for p in prep.values() for t in (p.get("foreign") or [])}) \
            or rec["foreign_tfs"]
        edits = {l: prep[l]["path"] for l in LAYERS if prep[l].get("path")}
        # Additive: a run whose SE -> closest-gene edits could not be built keeps both measured ends.
        se_edits = {l: prep[l]["path"] for l in SE_LAYERS if (prep.get(l) or {}).get("path")}
        if not edits:
            rec["status"], rec["reason"] = "failed", "; ".join(
                "%s end: %s" % (l, prep[l].get("reason") or "no edits") for l in LAYERS)
            for arm in ARMS:
                rec["arms"][arm] = {"arm": arm, "status": "skipped", "error": rec["reason"],
                                    "reason": "no edits were built: " + rec["reason"],
                                    "layers": {}, "rows": {l: 0 for l in LAYERS}}
            return self._write(rec, axis_path, t0)

        # The lineage-foreign null IS the calibration: beats() needs at least three foreign rows in
        # a group, so without the panel nothing can pass and every model set would read 0/n, an
        # infinite odds ratio and p 1 -- a calibrated-looking negative where nothing was tested.
        # Decided here, BEFORE the arms, so an uncalibratable run spends no quota and no GPU.
        if not rec["foreign_tfs"]:
            # The bare rule reason, then the fallback ONCE: notes["neg_tfs"] already ends with
            # C.fallback("neg_tfs"), so using it here printed the sentence twice (review 3, #15).
            why = ("no lineage-foreign null for this run: %s -- %s"
                   % (cfg.get("foreign_source") or "no curated foreign panel for this lineage",
                      C.fallback("neg_tfs")))
            rec["status"], rec["reason"] = "failed", why
            rec["models_withheld"] = {m: why for m in HEADLINES}
            for arm in ARMS:
                rec["arms"][arm] = {"arm": arm, "status": "skipped", "error": why,
                                    "reason": "the axis test cannot be calibrated: " + why,
                                    "layers": {}, "rows": {l: 0 for l in LAYERS}}
            self._log("NOT CALIBRATABLE: " + why)
            return self._write(rec, axis_path, t0)

        paths = {arm: {l: str(out_dir / ("%s_%s.jsonl" % (arm, l)))
                       for l in LAYERS + SE_LAYERS}
                 for arm in ARMS}
        rec["result_files"] = paths
        rec["stale_moved"], rec["stand_in_cleared"], rec["edits_md5"] = {}, {}, {}
        for arm in ARMS:
            for layer in LAYERS + SE_LAYERS:
                _ed = edits.get(layer) or se_edits.get(layer)
                if not _ed:
                    continue
                moved, md5 = _fingerprint(paths[arm][layer], _ed, log=self._log)
                rec["edits_md5"][layer] = md5
                if moved:
                    rec["stale_moved"]["%s/%s" % (arm, layer)] = moved
                gone = _clear_stand_in(paths[arm][layer])
                if gone:
                    rec["stand_in_cleared"]["%s/%s" % (arm, layer)] = gone
                    self._log("removed an empty stand-in left at %s/%s (%s): it would make this "
                              "re-run unusable" % (arm, layer, ", ".join(gone)))

        self._run_arms(rec, edits, paths, run_out, agent_root, se_edits=se_edits)
        self._score(rec, out_dir, prep, min_rd)
        return self._write(rec, axis_path, t0)

    def _trn_input(self, C, run_out):
        """(is this run's own TRN the one that would be read, the reason it is not).

        The runner names se_cyjs_json as an input.  A missing KEY means nobody passed one (a
        direct call, or the dry harness), and the step derives it as before; an EMPTY value means
        the runner had none to give, because the se_trn step failed or was skipped.
        """
        given = self.inputs.get("se_cyjs_json", None)
        if given is None:
            return True, None
        given = str(given).strip()
        if not given:
            return False, ("this run built no TRN: the runner handed the step no se_cyjs.json "
                           "(the se_trn step failed or was skipped), so there are no axes of THIS "
                           "run to validate.  An earlier run's TRN left in the output directory "
                           "is not this run's and is not validated in its place.")
        used = (C.manifest(run_out).get(("se_trn", "se_cyjs_json"))
                or os.path.join(run_out, "se_trn", "se_cyjs.json"))
        if os.path.realpath(used) != os.path.realpath(given):
            return False, ("the TRN this step would read (%s) is not the one the runner built in "
                           "this run (%s), so the axes would be validated against a network this "
                           "run did not build" % (used, given))
        return True, None

    def _skip(self, rec, axis_path, why, t0):
        """A skip the report can state: the step did not run, and this is why."""
        rec["status"], rec["reason"] = "skipped", why
        for arm in ARMS:
            rec["arms"][arm] = {"arm": arm, "status": "skipped", "error": why,
                                "reason": "the step did not run: " + why,
                                "layers": {}, "rows": {l: 0 for l in LAYERS}}
        self._log("SKIPPED: " + why)
        return self._write(rec, axis_path, t0)

    def _build_edits(self, out_dir, run_out, agent_root, trn_order, target_order):
        """Both edit layers, written to this step's own directory.  The arms replay these files,
        so their status is recorded per layer rather than raised: a target end that cannot be
        built still leaves the enhancer end, which is the end the evidence is at."""
        from mrag_axis.prep import edits_enhancer as PE, edits_target as PT
        from mrag_axis.prep import edits_se_direct as SD
        p = self.params
        jobs = {
            "enhancer": (PE, dict(n_se=int(p.get("n_se", 40)), n_ctl=int(p.get("n_ctl", 40)),
                                  score_frac=float(p.get("score_frac", 0.85)),
                                  max_motifs=int(p.get("max_motifs", 40)),
                                  trn_order=trn_order, agent_root=agent_root)),
            "target": (PT, dict(max_peaks_per_gene=int(p.get("max_peaks_per_gene", 3)),
                                score_frac=float(p.get("score_frac", 0.85)),
                                max_motifs=int(p.get("max_motifs", 40)),
                                trn_order=trn_order, target_order=target_order,
                                agent_root=agent_root)),
            # The SE -> closest-gene end needs no factor list and no promoter peak, so it takes
            # nothing but the run: its items are the TRN's own SE -> closest-gene edges.
            "se_direct": (SD, dict(agent_root=agent_root)),
            "se_near": (_SeNearPrep, dict(agent_root=agent_root)),
        }
        out = {}
        for layer, (mod, kw) in jobs.items():
            try:
                res = mod.build(run_out, **kw)
            except Exception as e:         # noqa: BLE001 -- build() returns a status; pyfaidx,
                out[layer] = {"status": "failed", "path": "",   # numpy or a missing hg38 would
                              "reason": "%s: %s" % (type(e).__name__, e)}
                self._log("%s-end edits FAILED: %s" % (layer, out[layer]["reason"]))
                continue
            path = ""
            if res.get("items"):
                path = str(out_dir / ("edits_%s.json" % layer))
                with open(path, "w") as fh:
                    # published_form(): the four keys the research prep wrote, which is the shape
                    # both arms and mrag_axis.stat.stats read.
                    json.dump(mod.published_form(res), fh)
            out[layer] = {"status": res.get("status"), "reason": res.get("reason"), "path": path,
                          "n_items": len(res.get("items") or []), "trn": res.get("trn") or [],
                          "foreign": res.get("foreign") or [], "detail": res.get("detail") or {}}
            self._log("%s-end edits: %s, %d items, %d pairs"
                      % (layer, out[layer]["status"], out[layer]["n_items"],
                         (out[layer]["detail"] or {}).get("n_pairs") or 0))
        return out

    def _run_arms(self, rec, edits, paths, run_out, agent_root, se_edits=None):
        """Both arms at once, each writing its own resumable JSONL pair.

        CONCURRENT because the two arms contend for nothing: the AlphaGenome arm is network-bound
        (quota and latency) and the Corgi arm GPU-bound behind its own lease, measured at ~66 and
        ~86 min on the published PRAD run -- serial is ~2.5 h of the pipeline's wall clock,
        parallel ~90 min.  Threads, not processes, because each arm only waits: both spend their
        time inside a child interpreter (the SDK's venv for AlphaGenome, the Corgi venv for
        torch), which is also why neither arm's imports have to exist in this one.
        """
        p = self.params
        jobs = {
            "alphagenome": (run_alphagenome, dict(
                tries=int(p.get("ag_tries", 6)), base_wait=float(p.get("ag_base_wait", 5.0)),
                **({"child_python": str(p["ag_python"])} if p.get("ag_python") else {}),
                **({"timeout_s": float(p["ag_timeout_s"])} if p.get("ag_timeout_s") else {}))),
            "corgi": (run_corgi, dict(
                **({"need_gib": float(p["need_gib"])} if p.get("need_gib") else {}),
                **({"lock_dir": str(p["lock_dir"])} if p.get("lock_dir") else {}),
                **({"timeout_s": float(p["corgi_timeout_s"])} if p.get("corgi_timeout_s")
                   else {}))),
        }
        with ThreadPoolExecutor(max_workers=len(ARMS)) as ex:
            futs = {arm: ex.submit(fn, edits, paths[arm], run_out, agent_root,
                                   log=self._arm_log(arm), **kw)
                    for arm, (fn, kw) in jobs.items()}
            for arm, f in futs.items():
                try:
                    rec["arms"][arm] = f.result()
                except Exception as e:     # noqa: BLE001 -- the arm is a WARN gate, never an
                    rec["arms"][arm] = {   # exception (the model-arm failure policy)
                        "arm": arm, "layers": {},
                        "reason": "the %s arm raised: %s: %s" % (arm, type(e).__name__, e)}
        for arm in ARMS:
            a = rec["arms"][arm]
            st, why = arm_status(a)
            # An arm that RAISED before writing any layer record stored the real exception in
            # a["reason"], and arm_status() then returns the generic "the arm produced no layer
            # record" -- which is never empty, so the cause used to be dropped from the axis JSON,
            # the gate and the report.  The stored reason wins whenever there is no layer record.
            a["reason"] = ((a.get("reason") or why or "") if not (a.get("layers") or {})
                           else (why or a.get("reason") or ""))
            a["status"] = st
            # The contract's per-arm record: status + error, so the report can say which arm is
            # missing and why without reading the arm's own log.
            a["error"] = "" if st == "ok" else a["reason"]
            a["rows"] = {l: _n_rows(paths[arm][l]) for l in LAYERS}
            self._log("%s arm: %s%s (%s)"
                      % (arm, st, (" -- " + a["reason"]) if a["reason"] else "",
                         ", ".join("%s %d rows" % (l, a["rows"][l]) for l in LAYERS)))
        # The SE -> closest-gene end, AFTER the two measured ends and in its own record: the Corgi
        # lease is taken again rather than held across both passes, and arm_status() never sees
        # these layers, so this end cannot withhold a model set (it is descriptive).
        if se_edits:
            with ThreadPoolExecutor(max_workers=len(ARMS)) as ex:
                futs = {arm: ex.submit(fn, se_edits, paths[arm], run_out, agent_root,
                                       layers=SE_LAYERS, log=self._arm_log(arm), **kw)
                        for arm, (fn, kw) in jobs.items()}
                for arm, f in futs.items():
                    try:
                        rec["arms"][arm]["se_layers"] = (f.result().get("layers") or {})
                    except Exception as e:     # noqa: BLE001 -- descriptive end, never an exception
                        rec["arms"][arm]["se_layers"] = {
                            l: {"status": "failed",
                                "reason": "the %s arm raised on the SE -> closest-gene end: %s: %s"
                                          % (arm, type(e).__name__, e)} for l in SE_LAYERS}
            for arm in ARMS:
                a = rec["arms"][arm]
                a["se_rows"] = {l: _n_rows(paths[arm][l]) for l in SE_LAYERS}
                self._log("%s arm, SE -> closest-gene end: %s (%s)"
                          % (arm, ", ".join("%s %s" % (l, (a["se_layers"].get(l) or {}).get("status"))
                                            for l in SE_LAYERS),
                             ", ".join("%s %d rows" % (l, a["se_rows"][l]) for l in SE_LAYERS)))

    def _score(self, rec, out_dir, prep, min_rd):
        """Score the stored per-pair results with definition="unified" -- report.by_model()'s
        three model sets, for the ones this run's arms can actually support.

        A model set whose arm wrote no rows is WITHHELD with its reason, not scored as zero:
        mrag_axis.stat.stats.beats() indexes the read-out column by name, and 'both models' in
        particular is the same read-out passing under both models in the same row, so with one arm
        missing it does not exist (PI decision: the consensus cannot exist without both arms).
        """
        from mrag_axis.stat import stats as S
        net = str(out_dir / "net_edges.json")
        with open(net, "w") as fh:
            # cfg["net"] for mrag_axis.stat.stats._grid: the run's own se_cyjs edge list, which
            # reproduces the research net_edges_*.json files exactly (config.net_edges).
            json.dump(rec["config"]["net_edges"], fh)
        agp = [rec["result_files"]["alphagenome"][l] for l in LAYERS]
        cells, cells_label, cells_why = _readout_cells(S, agp, rec.get("target_cells_declared"))
        scfg = {
            # NOT "oc2" (that branch reads the research seko2 TSV) and NOT "prad" until the AR
            # read-outs are shown to exist: branch=="prad" is what adds the AR-gated row, and the
            # AR gate is a declared extra requirement, never a default (PI decision 8).
            "branch": "run",
            "enh_edits": prep["enhancer"]["path"],
            "enh_ag": rec["result_files"]["alphagenome"]["enhancer"],
            "enh_cg": rec["result_files"]["corgi"]["enhancer"],
            "enh_ag_pub": rec["result_files"]["alphagenome"]["enhancer"],
            "enh_pub": None, "seko2": None, "axis_pub": None,
            "edits": prep["target"]["path"],
            "ag": rec["result_files"]["alphagenome"]["target"],
            "cg": rec["result_files"]["corgi"]["target"],
            "tgt": cells, "net": net,
        }
        rec["read_outs"] = {"enhancer": list(S.ENH_RD), "target": list(S.TGT_RD),
                            "min_rd": min_rd, "cells": cells_label, "reason": cells_why,
                            "cells_declared": rec.get("target_cells_declared") or "",
                            "trn_order": rec.get("trn_order"),
                            "target_order": rec.get("target_order"),
                            "enhancer_tracks": {"h3k27ac": S.TRACKS["h3k27ac_prostate"],
                                                "dnase": S.TRACKS["dnase_prostate"]},
                            "target_tracks": {rd: [a[0], a[1], c[0], c[1]]
                                              for rd, (a, c) in cells.items()}}
        # The biosamples the AlphaGenome arm ACTUALLY read, not the catalogue it asked for: the
        # request is the whole lineage group, and the section's intro has to name what was read
        # out (the contract: "the read-out biosamples of THIS run").
        req_bios = (rec["arms"].get("alphagenome") or {}).get("biosamples") or {}
        # Provisional (the enhancer read-outs); narrowed to what was really scored below.
        rec["biosamples"] = _read_biosamples(req_bios, agp)
        rec["biosamples_requested"] = dict(req_bios)
        try:
            rec["dropped_pairs"] = [list(x) for x in
                                    sorted(S.ctl_in_readout_bin(scfg["enh_edits"], "region"))]
        except Exception as e:             # noqa: BLE001 -- descriptive; a missing edits file
            rec["notes_dropped_pairs"] = "%s: %s" % (type(e).__name__, e)
        if not scfg["enh_edits"]:
            # The enhancer end IS the evidence (PI decision 1), so without it there is nothing to
            # report; the target end alone is descriptive.
            rec["status"], rec["reason"] = "failed", "; ".join(
                [x for x in (rec["reason"],
                             "the enhancer end has no edits, and it is the end the evidence rests "
                             "on: %s" % (prep["enhancer"].get("reason") or "no items")) if x])
            return
        # A target end that could not be built does NOT discard the measured enhancer end: the
        # target end is descriptive (PI decision 1), so the enhancer end is scored alone and the
        # target rows and the triples are withheld with this reason.
        ends = "both" if scfg["edits"] else "enhancer"
        if ends == "enhancer":
            rec["target_withheld"] = (
                "the target end was not scored: %s. The enhancer end, which is the end the "
                "evidence rests on, is reported in full; the SE-TF-target triples need both ends "
                "and are not drawn." % (prep["target"].get("reason") or "no target-end edits"))
            self._log("target end withheld: " + rec["target_withheld"])
        try:
            e, _dropped = S.enh_frame(None, definition="unified", excl="rule", cfg=scfg)
            b = (None if ends == "enhancer"
                 else S.tgt_frame(None, definition="unified", src="all", cfg=scfg))
        except Exception as ex:            # noqa: BLE001 -- unreadable results are a reason
            rec["status"], rec["reason"] = "failed", (
                "the per-pair results could not be read: %s: %s" % (type(ex).__name__, ex))
            return
        enh_ok = _models_with_columns(e, S.ENH_RD)
        tgt_ok = enh_ok if b is None else _models_with_columns(b, S.TGT_RD)
        # PER MODEL which ends exist (review 3, #0): a model with its enhancer columns is scored;
        # one that also lacks its target columns keeps its enhancer end and has only its target
        # row and triples withheld (score_models' usable_target).
        usable = [m for m in ("ag", "corgi") if m in enh_ok]
        usable_tgt = [m for m in usable if m in tgt_ok]
        rec["frames"] = {"enhancer_rows": int(len(e)),
                         "target_rows": (None if b is None else int(len(b))),
                         "models_enhancer": enh_ok,
                         "models_target": (None if b is None else tgt_ok), "ends": ends}
        ar = False
        if (rec.get("ar_gate_declared") and "ag" in usable
                and all(("ag_" + k) in getattr(e, "columns", []) for k in ("ar22", "arvc"))):
            # DECLARED (run.toml axis_ar_gate), never a default: every prostate-lineage run's
            # AlphaGenome request carries the 22Rv1 / VCaP tracks, so switching the row on
            # whenever those columns exist turned the PRAD-only extra requirement into the Holm
            # headline of every prostate run, LNCaP/OC2 included -- where the enhancer row is the
            # one that leads (PI decision 8 and the OC2 headline rule).  It is also an ALPHAGENOME
            # read-out, so it is not reported at all when that arm is withheld.
            scfg["branch"], ar = "prad", True
            rec["read_outs"]["ar_gate"] = [S.TRACKS["h3k27ac_22Rv1"], S.TRACKS["h3k27ac_VCaP"]]
        elif rec.get("ar_gate_declared"):
            rec["read_outs"]["ar_gate_reason"] = (
                "the AR cross-condition requirement is declared for this run but its AlphaGenome "
                "read-outs (22Rv1 and VCaP H3K27ac) are not on disk, so the AR row is not "
                "reported")
        # A half-measured arm must not shrink the other arm's denominators: at an end where an
        # arm is incomplete, each single-model set is scored on its OWN rows there and only the
        # consensus is intersected across both (see _model_cfg -- per end and per arm, so a
        # complete AlphaGenome enhancer layer still gives the Corgi set its declared AR row).
        incomplete = _incomplete_layers(rec["arms"])
        rec["own_rows"] = bool(incomplete["enhancer"] or incomplete["target"])
        rec["own_rows_by_end"] = {l: [ARM_OF[m] for m in incomplete[l]] for l in LAYERS}

        def arm_why(m):
            return (rec["arms"].get(ARM_OF.get(m, m)) or {}).get("reason") or \
                "no read-out column on disk"

        def tgt_why(m):
            lay = (((rec["arms"].get(ARM_OF.get(m, m)) or {}).get("layers") or {})
                   .get("target") or {})
            return ("target end: %s" % lay["reason"]) if lay.get("reason") else arm_why(m)

        notes = {}
        try:
            rec["models"], rec["models_withheld"] = score_models(
                scfg, e, b, usable, min_rd, own_rows=incomplete, ends=ends, ar_gate=ar,
                why_missing=arm_why, usable_target=usable_tgt, target_why=tgt_why, notes=notes)
        except Exception as ex:            # noqa: BLE001 -- a scoring failure is a reported
            rec["models"], rec["models_withheld"] = {}, {}   # status and a WARN gate, never a
            rec["status"], rec["reason"] = "failed", (        # failed step (PI decision 3)
                "the stored per-pair results could not be scored: %s: %s"
                % (type(ex).__name__, ex))
        rec["targets_withheld"] = notes.get("targets_withheld") or {}
        rec["ar_withheld"] = notes.get("ar_withheld") or {}
        rec["headline_rows"] = notes.get("headline_rows") or {}
        rec["se_tf_edges"] = notes.get("se_tf_edges") or {}
        for name, why in rec["targets_withheld"].items():
            self._log("%s set: %s" % (name, why))
        # The biosamples the statistic READ OUT (review 3, #4): the enhancer read-outs, the
        # target read-out cells when an AlphaGenome target row was scored, and the AR tracks when
        # the declared gate is on -- each only when its track is on disk.
        scored = [S.TRACKS["h3k27ac_prostate"], S.TRACKS["dnase_prostate"]]
        if any("pair_target" in rec["models"].get(n, {}) for n in ("alphagenome", "both")):
            scored += [a[0] for (a, _c) in cells.values()]
        if ar:
            scored += [S.TRACKS["h3k27ac_22Rv1"], S.TRACKS["h3k27ac_VCaP"]]
        rec["read_outs"]["scored_tracks"] = scored
        rec["biosamples"] = (_read_biosamples(req_bios, agp, scored)
                             if any("ag" in S.MODELS[n] for n in rec["models"]) else {})
        rec["arm_effects"] = _arm_effects(rec)
        rec["se_gene"] = self._score_se_gene(rec, prep, cells)

    def _score_se_gene(self, rec, prep, cells):
        """The SE -> closest-gene end, scored from its own stored predictions.

        Descriptive (PI decision 2026-10-20), so every failure is a reason in this record and
        nothing here can change rec["models"], rec["status"] or the Holm family.
        """
        from mrag_axis.stat import se_direct as SDS
        try:
            fp = (prep.get("se_direct") or {}).get("path")
            np_ = (prep.get("se_near") or {}).get("path")
            if not fp:
                return {"status": "skipped", "rows": [], "summary": {}, "strict": [],
                        "reason": ((prep.get("se_direct") or {}).get("reason")
                                   or "no SE -> closest-gene edits were built")}
            free = json.load(open(fp))
            near = json.load(open(np_)) if np_ and os.path.exists(np_) else {"items": [], "roles": {}}
            res = {(m, fam): (rec.get("result_files") or {}).get(ARM_OF[m], {}).get(l)
                   for m in ("ag", "corgi") for fam, l in (("free", "se_direct"), ("near", "se_near"))}
            ro = {"ag": {rd: (a[0], a[1]) for rd, (a, _c) in cells.items()},
                  "corgi": {rd: (c[0], c[1]) for rd, (_a, c) in cells.items()}}
            out = SDS.score(free, near, res, ro)
            self._log("SE -> closest-gene end: %s -- %s"
                      % (out.get("status"),
                         "; ".join("%s %s: free %d/%d, near %d/%d"
                                   % (m, rd, v["free_pass"], v["n"], v["near_pass"], v["n_near"])
                                   for m, d in (out.get("summary") or {}).items()
                                   for rd, v in d.items() if rd in SDS.REPORTED) or "nothing scored"))
            return out
        except Exception as ex:                # noqa: BLE001 -- a descriptive end states its reason
            return {"status": "failed", "rows": [], "summary": {}, "strict": [],
                    "reason": "the SE -> closest-gene end could not be scored: %s: %s"
                              % (type(ex).__name__, ex)}

    # ---------------------------------------------------------------- emit
    def _finish_skipped(self, rec, axis_path, blocked, C, t0):
        """The step cannot run at all.  It records a skip and the consequence, and the report
        states it -- the same shape the runner's organism gate writes for a non-human run."""
        rec["status"] = "skipped"
        rec["reason"] = "; ".join("%s: %s -- %s" % (k, rec["config"]["missing"][k], C.fallback(k))
                                  for k in blocked)
        for arm in ARMS:
            rec["arms"][arm] = {"arm": arm, "status": "skipped", "error": rec["reason"],
                                "reason": "the step did not run: " + rec["reason"],
                                "layers": {}, "rows": {l: 0 for l in LAYERS}}
        self._log("SKIPPED: " + rec["reason"])
        return self._write(rec, axis_path, t0)

    def _write(self, rec, axis_path, t0):
        """Write axis_<domain>.json and set the outputs.  The JSON is written whatever the status:
        a section that has to state why an arm is missing needs the record, not an absent file."""
        n_axes = max([0] + [(m.get("triples") or {}).get("axes") or 0
                            for m in rec["models"].values()])
        rec["default_model"], rec["headline_reason"] = choose_headline(
            rec["models"], rec.get("headline_declared"))
        if rec["status"] == "ok":
            if not rec["models"]:
                rec["status"], rec["reason"] = "failed", (
                    "no model set could be scored: "
                    + "; ".join("%s (%s)" % (k, v) for k, v in rec["models_withheld"].items()))
            elif (rec["models_withheld"] or rec.get("target_withheld")
                  or rec.get("targets_withheld")
                  or any(a.get("status") != "ok" for a in rec["arms"].values())):
                rec["status"] = "partial"
                rec["reason"] = "; ".join(
                    ["%s arm: %s" % (a, rec["arms"][a].get("reason"))
                     for a in ARMS if rec["arms"][a].get("status") != "ok"]
                    + ["%s withheld: %s" % (k, v) for k, v in rec["models_withheld"].items()]
                    + ([rec["target_withheld"]] if rec.get("target_withheld") else [])
                    + ["%s target end withheld" % k
                       for k in (rec.get("targets_withheld") or {})
                       if not rec.get("target_withheld")])
        for arm, eff in (rec.get("arm_effects") or {}).items():
            if arm in rec["arms"]:
                rec["arms"][arm]["effect"] = eff
        rec["elapsed_s"] = round(time.time() - t0, 2)
        Path(axis_path).parent.mkdir(parents=True, exist_ok=True)
        # The per-arm record on its own, so a re-render or a hand check can read which arm is
        # missing and why without parsing the whole statistic (the Corgi arm writes its own
        # <out>.status.json too; this is both arms, after the node's own classification).
        with open(Path(axis_path).parent / "arms_status.json", "w") as fh:
            json.dump({"domain": rec["domain"], "status": rec["status"],
                       "reason": rec["reason"], "arms": rec["arms"]}, fh, indent=1, default=str)
        with open(axis_path, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        self.outputs = {
            "axis_json": str(axis_path),
            "axis_alphagenome": {k: (rec["arms"].get("alphagenome") or {}).get(k)
                                 for k in ("status", "reason", "error", "rows", "effect")},
            "axis_corgi": {k: (rec["arms"].get("corgi") or {}).get(k)
                           for k in ("status", "reason", "error", "rows", "effect")},
            "axis_prep": {l: {k: (rec["prep"].get(l) or {}).get(k)
                              for k in ("status", "reason", "n_items")} for l in LAYERS},
            "axis_step": {"status": rec["status"], "reason": rec["reason"],
                          "domain": rec["domain"], "trn_order": rec.get("trn_order", ""),
                          "target_order": rec.get("target_order", ""),
                          "headline": rec["default_model"],
                          "headline_reason": rec["headline_reason"],
                          "models": sorted(rec["models"]),
                          "withheld": dict(rec["models_withheld"]),
                          "elapsed_s": rec["elapsed_s"]},
            "n_axes": int(n_axes),
            "n_models": len(rec["models"]),
        }
        self._log("axis_%s.json: %s -- %d model set(s), %d axes%s"
                  % (rec["domain"], rec["status"], len(rec["models"]), n_axes,
                     (" -- " + rec["reason"]) if rec["reason"] else ""))

    # ---------------------------------------------------------------- gates
    def validate(self):
        """Post-flight gates.  NONE of them can be "fail": a failed arm is a WARNING and the run
        continues (PI decision 3, the model-arm failure policy applied to both arms)."""
        gates = []
        st = (self.outputs or {}).get("axis_step") or {}
        ax = (self.outputs or {}).get("axis_json") or ""
        ok = bool(ax) and Path(ax).exists()
        # The gate is on the STEP'S STATUS, not merely on the file's existence: the axis JSON is
        # written whatever happens (a section that has to state why a number is missing needs the
        # record), so "the file exists" passed even when nothing could be tested -- a scoring
        # failure with both arms ok got no warning at all, and the report called it an empty
        # result.  Still never "fail": a failure here is a warning and the run continues.
        good = st.get("status") in ("ok", "partial")
        gates.append({"name": "axis_json", "result": "pass" if (ok and good) else "warn",
                      "value": int((self.outputs or {}).get("n_models") or 0),
                      "msg": ("axis_%s.json: %s, %s model set(s), %s axes"
                              % (st.get("domain"), st.get("status"),
                                 (self.outputs or {}).get("n_models"),
                                 (self.outputs or {}).get("n_axes")) if (ok and good)
                              else ("axis_%s.json: the axis test is %s -- %s"
                                    % (st.get("domain"), st.get("status"),
                                       str(st.get("reason") or "no reason recorded")[:300])
                                    if ok else "no axis JSON was written (%s)"
                                    % str(st.get("reason") or "")[:300]))})
        for arm in ARMS:
            a = (self.outputs or {}).get("axis_" + arm) or {}
            if a.get("status") in (None, "ok"):
                continue
            # What this arm's state did to the other numbers, as scored (_arm_effects); a step
            # that never reached scoring says only that the section states what is missing.
            eff = a.get("effect") or ("nothing was scored for this arm; the section states "
                                      "which arm is missing and why")
            gates.append({"name": "axis_" + arm, "result": "warn", "value": 0,
                          "msg": ("the %s arm of Section 13 is %s: %s -- %s"
                                  % (arm, a.get("status"), str(a.get("reason") or "")[:300],
                                     str(eff)[:600]))})
        if st.get("withheld"):
            gates.append({"name": "axis_models", "result": "warn", "value": len(st["withheld"]),
                          "msg": ("Section 13 withholds %d of 3 model set(s): %s"
                                  % (len(st["withheld"]),
                                     "; ".join("%s (%s)" % (k, str(v)[:160])
                                               for k, v in st["withheld"].items())))})
        return gates

    def to_manifest(self) -> dict:
        rec = super().to_manifest()
        st = (self.outputs or {}).get("axis_step") or {}
        if rec.get("status") == "ok" and st.get("status") == "skipped":
            # The step could not run at all (no TRN / no ROSE table).  The manifest says skipped
            # with the reason -- the shape the runner's organism gate writes -- so the report can
            # state it instead of showing an empty section.
            rec["status"] = "skipped"
            rec["note"] = st.get("reason") or ""
        return rec
