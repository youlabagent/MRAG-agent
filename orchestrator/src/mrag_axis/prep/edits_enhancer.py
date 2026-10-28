"""The enhancer-end edit items: one RNG stream, replayed in the research order.

Productionised from /BIO1/MRAG/corgi/ft/ag/agprep_A.py and ft/prad/agprep_A_prad.py, whose shared
docstring states the requirement this module exists to meet: it "replays [seko2's] random-number
stream in the same order, so every shuffled motif and every control position is bit-identical to
the Corgi run".  That is why nothing here may be reordered, vectorised or parallelised.

The stream, in the order the draws are consumed, is the whole contract:

  for each region row, in pick_regions() order (40 SE by ascending rank, then the ordinary ones)
      a region skipped by the window test consumes NOTHING
      for each factor, in (trn + foreign) order
          no motif hit  -> consumes nothing, and the factor is absent from the item
          >max_motifs   -> ONE rng.choice(n, max_motifs, replace=False)
          per kept hit, in ascending position:  shuffle(KO bases)
                                                rng.integers(1000, WINDOW-L-1000) until the draw
                                                  lands outside the region  (every rejected draw
                                                  is consumed too)
                                                shuffle(control bases)

Two things the research hardcoded are derived here instead (the point of the exercise):
the run's output directory is an argument, and the ROSE table, its signal column, the TRN factor
list, the lineage-foreign null and the SE->TF assignment all come from that run through mrag_axis.stat.config.

ONE field is not derivable and is a parameter: `trn_order`.  See TRN_ORDER below.
"""
from __future__ import annotations
import hashlib

import numpy as np

from mrag_axis.stat import config as C
from .motifs import JASPAR_MEME, load_pwms, sample_hits, scan
from .regions import pick_regions

HG38_FA = f"{C.AGENT_ROOT}/Reference/ATAC_hg38/hg38.fa"

# The factor iteration order fixes which draws of the single stream go to which (region, factor),
# so it has to match the research script to reproduce its edits -- and the two research scripts
# disagree.  seko2_prad.TRN_TFS is the run's se_cyjs.json TF-node order (what mrag_axis.stat.config.
# trn_tf_names returns, "network"); seko2.TRN_TFS is the same set typed alphabetically ("sorted").
# Neither is recoverable from the run, so this is a declared parameter: "network" is the default
# because it is the one order a node can derive, and it is the one the later (PRAD) script used.
TRN_ORDER = ("network", "sorted")


def trn_and_foreign(run_output, trn_order="network", trn_tfs=None, foreign=None,
                    jaspar=JASPAR_MEME, agent_root=C.AGENT_ROOT):
    """(trn, foreign, no_pwm, status) -- the two factor lists, in iteration order.

    trn: this run's TRN factors that have a JASPAR MOTIF (mrag_axis.stat.config.trn_tf_names + split_pwm),
    ordered by `trn_order` -- an explicit `trn_tfs` list is taken in the order given instead.  foreign: mrag_axis.stat.config.neg_tfs, i.e. FOREIGN_PANEL[lineage] minus this
    lineage's own factors and anything already in the TRN -- the eight-factor null seko2's
    docstring argues for, which is what separates "panel motifs matter" from "motifs matter".
    """
    st = {"status": "ok"}
    if trn_tfs is not None:
        keep, drop = list(trn_tfs), []
        st["trn_source"] = "caller"
    else:
        names, why = C.trn_tf_names(run_output)
        if not names:
            return None, None, None, {"status": "skipped", "reason": f"no TRN factors: {why}",
                                      "consequence": C.fallback("net_edges")}
        keep, drop = C.split_pwm(names, jaspar=jaspar)
        st["trn_source"] = f"se_cyjs TF nodes ({len(names)} names)" + (f"; {why}" if why else "")
        if trn_order == "sorted":
            keep = sorted(keep)
        elif trn_order != "network":
            if isinstance(trn_order, str):
                return None, None, None, {"status": "failed",
                                          "reason": f"trn_order must be one of {TRN_ORDER} or an "
                                                    f"explicit list, got {trn_order!r}"}
            order = [str(t).upper() for t in trn_order]
            keep = sorted(keep, key=lambda t: order.index(t.upper()) if t.upper() in order
                          else len(order))
            st["trn_source"] += "; order supplied by the caller"
    st["trn_order"] = trn_order if isinstance(trn_order, str) else list(trn_order)
    st["no_pwm"] = drop
    if foreign is not None:
        neg, why = list(foreign), "caller"
    else:
        neg, why = C.neg_tfs(run_output, keep, agent_root=agent_root, jaspar=jaspar)
    if not neg:
        # The null is the calibration, not a nicety: without it the axis count is uncalibrated.
        # `st` already carries status "ok", so it goes FIRST: spread last it would put "ok"
        # back over "partial" and build() would never report the missing null.
        st = {**st, "status": "partial", "reason": f"no lineage-foreign null: {why}",
              "consequence": C.fallback("neg_tfs")}
        neg = []
    st["foreign_source"] = why
    return keep, neg, drop, st


def assigned_pairs(run_output):
    """{(SE label, factor)} that the TRN itself links -- seko2's test 2, "assigned vs unassigned".

    The research scripts read a hand-built ft/data/se_tf_assign.tsv.  That table IS the run's own
    SE_TF edge list: mrag_axis.stat.config.net_edges(out)['se_tf'] reproduces all 183 OC2 pairs and all 180
    PRAD pairs with nothing on either side, so the file is derived here instead of shipped.
    """
    ne, why = C.net_edges(run_output)
    if not ne:
        return set(), f"no SE->TF edges: {why}"
    return {(se, tf) for se, tf in ne["se_tf"]}, None


def build(run, n_se=40, n_ctl=40, score_frac=0.85, max_motifs=40, seed=0,
          trn_order="network", trn_tfs=None, foreign=None, jaspar=JASPAR_MEME,
          genome_fa=HG38_FA, agent_root=C.AGENT_ROOT):
    """The enhancer-end edits for one run: {status, reason, window, trn, foreign, items}.

    `run` is a run id, a run directory or a run's output directory -- mrag_axis.stat.config.run_output
    resolves all three.  Returns a status instead of raising on every missing input, because the
    node this feeds runs by default with no on/off switch.
    """
    from pyfaidx import Fasta

    out = C.run_output(run, agent_root)
    G = C.corgi_channels()
    WINDOW, HALF, FLANK, NBIN, BIN = (G["WINDOW"], G["HALF"], G["FLANK"], G["NBIN"], G["BIN"])
    res = {"status": "ok", "reason": None, "window": WINDOW, "trn": [], "foreign": [], "items": []}

    trn, neg, no_pwm, tst = trn_and_foreign(out, trn_order=trn_order, trn_tfs=trn_tfs,
                                            foreign=foreign, jaspar=jaspar, agent_root=agent_root)
    if trn is None:
        return {**res, **{k: tst[k] for k in ("status", "reason")}, "detail": tst}
    reg, rst = pick_regions(out, n_se=n_se, n_ctl=n_ctl, seed=seed, agent_root=agent_root)
    if reg is None:
        return {**res, "status": rst["status"], "reason": rst["reason"], "detail": rst}
    assign, awhy = assigned_pairs(out)

    pwms = load_pwms(set(trn) | set(neg), jaspar=jaspar)
    trn = [t for t in trn if t in pwms]
    neg = [t for t in neg if t in pwms]
    res["trn"], res["foreign"] = trn, neg
    fa = Fasta(genome_fa)
    rng = np.random.default_rng(seed)                 # ONE stream for the whole build
    skipped = []

    for _i, r in reg.iterrows():
        mid = (int(r.start) + int(r.stop)) // 2
        ws = mid - HALF
        try:
            clen = len(fa[r.chrom])
        except Exception:
            # Research would have raised here; skipping consumes no draws, so the stream for every
            # other region is unchanged and the run still reports a section.
            skipped.append((r.label, f"{r.chrom} not in {genome_fa}"))
            continue
        if ws < 0 or ws + WINDOW > clen:
            skipped.append((r.label, "524-kb window runs off the chromosome"))
            continue
        wt = str(fa[r.chrom][ws:ws + WINDOW].seq).upper()
        cs = ws + FLANK
        sb, eb = max((int(r.start) - cs) // BIN, 0), min((int(r.stop) - cs) // BIN + 1, NBIN)
        if eb <= sb:
            skipped.append((r.label, "region outside the predicted central window"))
            continue
        lo_i, hi_i = int(r.start) - ws, int(r.stop) - ws
        rec = {"label": r.label, "cls": r.cls, "rank": int(r.enhancerRank), "signal": float(r.signal),
               "chrom": r.chrom, "ws": ws, "wt_md5": hashlib.md5(wt.encode()).hexdigest(),
               "windows": {"region": [[int(r.start), int(r.stop)]], "whole": [[ws, ws + WINDOW]]},
               "tfs": {}}
        for tf in trn + neg:
            hits = [(lo_i + p, L) for p, L in scan(wt[lo_i:hi_i], pwms[tf], score_frac)]
            if not hits:
                continue
            hits = sample_hits(hits, max_motifs, rng)
            ko_e, ct_e = [], []
            for pos, L in hits:
                sh = list(wt[pos:pos + L]); rng.shuffle(sh)
                ko_e.append([int(pos), "".join(sh)])
                while True:
                    q = int(rng.integers(1000, WINDOW - L - 1000))
                    if not (lo_i <= q <= hi_i):      # the control edit must miss the region
                        break
                sh = list(wt[q:q + L]); rng.shuffle(sh)
                ct_e.append([q, "".join(sh)])
            rec["tfs"][tf] = {"n_motifs": len(hits), "tf_class": "TRN" if tf in trn else "foreign",
                              "assigned": (r.label, tf) in assign, "ko": ko_e, "ctl": ct_e}
        res["items"].append(rec)

    res["detail"] = {"run_output": out, "regions": rst, "factors": tst, "no_pwm": no_pwm,
                     "assigned_pairs": len(assign), "assigned_note": awhy,
                     "skipped_regions": skipped, "seed": seed, "score_frac": score_frac,
                     "max_motifs": max_motifs, "jaspar": jaspar, "genome_fa": genome_fa,
                     "n_pairs": sum(len(x["tfs"]) for x in res["items"])}
    if tst["status"] != "ok":
        res["status"], res["reason"] = tst["status"], tst.get("reason")
    if awhy:
        res["status"] = "partial" if res["status"] == "ok" else res["status"]
        res["reason"] = awhy if res["reason"] is None else res["reason"]
    return res


def published_form(res):
    """The four keys agprep_A*.py wrote, in its key order, for the byte/field comparison."""
    return {"window": res["window"], "trn": res["trn"], "foreign": res["foreign"],
            "items": res["items"]}
