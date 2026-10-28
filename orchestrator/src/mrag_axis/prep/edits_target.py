"""The promoter/target-end edit items: one RNG stream, replayed in the research order.

Productionised from /BIO1/MRAG/corgi/ft/ag/agprep_D.py and its two siblings agprep_nec.py /
agprep_sclc.py (which differ from it in exactly two lines, the --out path and the literature
network file) plus /BIO1/MRAG/corgi/ft/prad/agprep_D_prad.py (which additionally swaps seko2 for
seko2_prad: a different run, a different TF list and a different HOMER peak table).  Four
published artefacts are the acceptance test, and equality with them is the whole point -- the
stored AlphaGenome and Corgi per-pair results are keyed to these exact edits, so a control
position off by one base is a different experiment, not a rounding difference:

  ft/ag/edits_D.json         OC2, prostate domain         24 network + 35 panel items
  ft/ag/edits_nec.json       OC2, NEC domain              27 network + 35 panel
  ft/ag/edits_sclc.json      OC2, SCLC domain             23 network + 35 panel
  ft/prad/edits_D_prad.json  PRAD, prostate domain        30 network + 46 panel

WHY THE CONTROL EDITS DIFFER FROM THE ENHANCER END'S.  Here the read-out is the target GENE -- RNA
over its exons, H3K4me3/CAGE at its TSS, DNase in the promoter peak -- so a control edit that
landed in an exon or in the promoter would move the read-out it exists to leave alone.
agprep_D.py's exclusion zone is therefore [the peak, TSS +/- 2000, every exon +/- 200], against
edits_enhancer's single [the region], and the draw is a bounded rejection loop (10000 attempts)
rather than edits_enhancer's unbounded `while True`, because three zones can in principle leave no
room.  Every rejected draw is consumed from the stream, so the loop bound is part of the contract.

THE ITEM SET has two sources and they are concatenated in this order, because the single RNG
stream walks the items in list order:

  network -- this run's own se_cyjs.json TF->target network (mrag_axis.stat.config.net_edges), one item per
             promoter peak of each target gene, at most `max_peaks_per_gene` peaks nearest the TSS
             from the run's HOMER peak table (mrag_axis.stat.config.peak_table).  The research read a
             pre-extracted ft/ag/net_edges_<domain>q38.json; that file IS the run's se_cyjs edge
             list (identical as a set on all four runs), so it is derived here instead of shipped.
  panel   -- stage 2a's mrag_axis.stat.panel.panel_items(), which rebuilds the old ag_validation node's item
             set from <run>/output/viz_built/consensus_ranked.tsv and already reproduces all 46
             PRAD and all 35 OC2 published panel items.  NOT curated_targets (2/46 and 6/35
             overlap only) and NOT ag_items.json, which stage 2c deletes with the node.
             The positional P_<gene>_<k> label scheme is kept exactly: the stored per-pair results
             are keyed by label, so renumbering silently decouples them from their measurement.

TWO inputs are not derivable from a run and are declared parameters:

  trn_order    the factor iteration order (see edits_enhancer.TRN_ORDER), which fixes which draws
               of the stream go to which (item, factor);
  target_order the network GENE order (TARGET_ORDER below), which fixes the network item order and
               therefore where in the stream each item's draws sit.

Nothing here calls AlphaGenome, loads a Corgi checkpoint or touches a GPU: it writes edits.
Needs numpy, pandas and pyfaidx.
"""
from __future__ import annotations
import collections
import hashlib
import math
import os
import re

import numpy as np

from mrag_axis.stat import config as C
from mrag_axis.stat import panel as PANEL

from .edits_enhancer import HG38_FA, trn_and_foreign
from .motifs import JASPAR_MEME, load_pwms, sample_hits, scan

# agprep_D.py's GTF.  refGene is the research choice and it matters: the TSS it yields is the TSS
# of a refSeq transcript, not of a GENCODE one, and the read-out windows are built around it.
REFGENE_GTF = f"{C.AGENT_ROOT}/Reference/ATAC_hg38/hg38.refGene.gtf"

# agprep_D.py's EXCL=2000 and its exon pad of 200: the half-width around the TSS and the pad around
# every exon that a control edit must clear.  These are the exclusion zone, not the read-out --
# prom1k/tss500 below are narrower on purpose, so the zone has margin around what is summed.
PROM_EXCL, EXON_PAD = 2000, 200

# agprep_D.py's window test: `ws + 1000 <= tss < ws + W - 1000`.  The read-out windows are built
# around the TSS, so a TSS in the outer kilobase would give a prom1k/tss500 interval clipped by
# the window edge and a read-out that is not comparable with the others.  The same 1000 is the
# lower bound of the control draw, rng.integers(1000, W - L - 1000).
EDGE_MARGIN = 1000

# The network GENE order.  "network" is se_cyjs.json's own Target-node order (what
# mrag_axis.stat.config.net_edges returns) and is the one a node can derive; "sorted" is that same set typed
# alphabetically.  The research pair disagrees: ft/prad/net_edges_prad_pc.json is byte-equal to the
# derived order, while ft/ag/net_edges_*q38.json carries the sorted one (same set, different
# order), and the published edits_D/nec/sclc items are in that sorted order.  An explicit sequence
# is also accepted and used verbatim for the genes it names.
TARGET_ORDER = ("network", "sorted")

_INDEX: dict = {}
_GENE_RE = re.compile(r'gene_name "([^"]+)"')
_TX_RE = re.compile(r'transcript_id "([^"]+)"')


def gene_index(gtf=REFGENE_GTF):
    """({symbol: {chrom: [transcript, ...]}}, status) from the refGene GTF's exon lines.

    agprep_D.py built this dict inline; the only thing that has to be preserved is the LIST ORDER
    per (symbol, chrom), because gene_model() resolves a tie for "nearest TSS" with min(), which
    keeps the first minimum.  The order is the order of first appearance of each transcript_id in
    the file, so the dict is filled exactly as the research filled it.

    Cached per path: the file is 286 MB / 836,541 exon lines and takes ~2.4 s to walk, and build()
    is called once per domain over the same genome.
    """
    if gtf in _INDEX:
        return _INDEX[gtf]
    tx = {}
    try:
        fh = open(gtf)
    except OSError as exc:
        # Never raise on a missing input: without a gene model there is no
        # target-end read-out at all, so the whole end is reported skipped with this reason.
        return None, {"status": "skipped", "reason": f"no refGene GTF: {exc}",
                      "consequence": C.fallback("refgene_gtf")}
    with fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            g, t = _GENE_RE.search(f[8]), _TX_RE.search(f[8])
            if not g or not t:
                continue
            tx.setdefault((g.group(1), f[0], t.group(1)),
                          {"strand": f[6], "exons": []})["exons"].append((int(f[3]) - 1, int(f[4])))
    idx: dict = {}
    for (sym, chrom, _tid), v in tx.items():
        v["start"] = min(e[0] for e in v["exons"])
        v["end"] = max(e[1] for e in v["exons"])
        idx.setdefault(sym, {}).setdefault(chrom, []).append(v)
    st = {"status": "ok", "gtf": gtf, "n_transcripts": len(tx), "n_symbols": len(idx)}
    _INDEX[gtf] = (idx, st)
    return _INDEX[gtf]


def tss_in_window(tss, ws, window):
    """agprep_D.py's item test: `ws + 1000 <= tss < ws + W - 1000`.

    Its own function because it is the one drop rule of this end that no published artefact
    exercises -- the window is centred on the peak and HOMER assigns every peak to its nearest
    gene, so on all four 2026-10 runs the tightest margin is 135,121 bp (edits_D.json) and
    261,191 bp (edits_D_prad.json), and raising max_peaks_per_gene to 200 does not move it.
    A guard that never fires on the data it was written for still has to be the research guard,
    so it is tested directly instead of being assumed.
    """
    return ws + EDGE_MARGIN <= tss < ws + window - EDGE_MARGIN


def gene_model(sym, chrom, centre, gtf=REFGENE_GTF, index=None):
    """TSS of the transcript nearest the peak, plus the MERGED exons and span of every same-strand
    transcript of that symbol on that chromosome (agprep_D.gene_model, verbatim).

    Nearest-to-the-peak rather than canonical because the item IS a peak: the promoter the peak
    belongs to is the one whose signal the edit perturbs.  Same-strand-only merging because the
    opposite strand at a bidirectional promoter is a different gene's read-out.
    """
    idx = index if index is not None else gene_index(gtf)[0]
    ts = (idx or {}).get(sym, {}).get(chrom)
    if not ts:
        return None
    tss = lambda t: t["start"] if t["strand"] == "+" else t["end"]
    best = min(ts, key=lambda t: abs(tss(t) - centre))
    same = [t for t in ts if t["strand"] == best["strand"]]
    merged = []
    for a, b in sorted(set(e for t in same for e in t["exons"])):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return {"tss": tss(best), "strand": best["strand"], "exons": merged,
            "body": [min(t["start"] for t in same), max(t["end"] for t in same)]}


def se_anno_path(run_output):
    """The run's se_anno.txt.  Rule: manifest.json step 'se_trn' output key 'se_anno_path', else
    the conventional se_trn/se_anno.txt -- the same two-step lookup mrag_axis.stat.config.net_edges uses for
    se_cyjs.json.  (mrag_axis.stat.config has no accessor of its own for this file; trn_tf_names inlines the
    conventional path as a fallback.)"""
    p = C.manifest(run_output).get(("se_trn", "se_anno_path")) or \
        f"{run_output}/se_trn/se_anno.txt"
    return (p, None) if os.path.exists(p) else (None, f"{p} missing")


def se_anno_observed(run_output):
    """({gene: (ATAC fold, RNA log2FC)}, reason) for the network targets, from se_anno.txt.

    agprep_D.py read `anno[anno['type'] == 'Target']` and kept
    (Fold(ATAC-seq), log2FC(RNA-seq)) per name, last row winning on a duplicate name.  A
    non-finite cell becomes None here, which is what the published records carry: json.dump would
    otherwise write a bare NaN that is not valid JSON and that compares unequal to itself.
    """
    import pandas as pd
    p, why = se_anno_path(run_output)
    if not p:
        # Observed fold-change is DESCRIPTIVE -- it is reported next to the item, never tested --
        # so a missing table costs those two fields and nothing else.
        return {}, f"no se_anno.txt ({why}); network items carry obs_log2fc/obs_atac_fold = None"
    anno = pd.read_csv(p, sep="\t")
    need = ["name", "type", "Fold(ATAC-seq)", "log2FC(RNA-seq)"]
    miss = [c for c in need if c not in anno.columns]
    if miss:
        return {}, f"{p} has no {miss}; network items carry obs_log2fc/obs_atac_fold = None"
    num = lambda v: None if v is None or not math.isfinite(float(v)) else float(v)
    out = {}
    for _i, r in anno[anno["type"] == "Target"].iterrows():
        out[r["name"]] = (num(r["Fold(ATAC-seq)"]), num(r["log2FC(RNA-seq)"]))
    return out, None


def homer_alias(tf_names, no_pwm=(), agent_root=C.AGENT_ROOT):
    """(alias map, reason) -- the ALIAS HALF ONLY of agprep_D.py's NAME map, {'GR','GRE'} ->
    NR3C1.

    mrag_axis.stat.config.name_map() derives the same two keys from production's HOMER_NAME_ALIASES for any
    run whose TRN contains NR3C1, but it also adds `no_pwm` -> None, and those entries must NOT be
    applied here: agprep_D.py's NAME had no null half, so mapping FOXM1 -> None would change which
    factors a panel peak claims.  Only the truthy entries are kept for that reason.
    """
    try:
        m = C.name_map(list(tf_names), list(no_pwm), agent_root=agent_root)
    except Exception as exc:
        return dict(PANEL.NAME), (f"mrag_axis.stat.config.name_map unavailable ({type(exc).__name__}); used "
                                  f"agprep_D.py's literal {sorted(PANEL.NAME)} -> NR3C1")
    return {a: s for a, s in m.items() if s}, None


def _ordered_targets(targets, target_order):
    """(gene list, reason) in the order the network items must be built in."""
    if target_order == "network":
        return list(targets), None
    if target_order == "sorted":
        return sorted(targets), None
    if isinstance(target_order, str):
        return None, f"target_order must be one of {TARGET_ORDER} or an explicit list, got " \
                     f"{target_order!r}"
    first = [str(g) for g in target_order]
    rest = [g for g in targets if g not in set(first)]
    return [g for g in first if g in set(targets)] + rest, "order supplied by the caller"


def network_items(run_output, max_peaks_per_gene=3, target_order="network", alias=None):
    """(items, status) -- one item per promoter peak of each TF->target gene of THIS run's TRN.

    The peaks are the `max_peaks_per_gene` rows of the run's HOMER peak table whose |Distance to
    TSS| is smallest, in that order, and the label is positional: N_<gene>_p<k>.  `claimed_tfs` is
    every factor the network links to the gene, sorted, so it does not depend on edge order.

    The peak is re-emitted as [Start - 1, End]: HOMER's table is 1-based inclusive and every
    coordinate in the edits JSON is half-open 0-based, the same convention mrag_axis.stat.panel.panel_items
    applies to the panel peaks.

    pandas.sort_values() is NOT stable (kind='quicksort'), so a |Distance to TSS| tie that
    straddles the head(max_peaks_per_gene) cut could in principle select a different peak -- and
    then the labels move with it.  Measured: 0 of the 80 target genes of the four 2026-10 runs has
    such a tie, so the cut is unambiguous on every published artefact.  Do not add kind='stable'
    without re-deriving the stored per-pair results.
    """
    import pandas as pd
    ne, why = C.net_edges(run_output)
    if not ne:
        return None, {"status": "skipped", "reason": f"no TF->target network: {why}",
                      "consequence": C.fallback("net_edges")}
    pk_path, pwhy = C.peak_table(run_output)
    if not pk_path:
        return None, {"status": "skipped", "reason": f"no HOMER peak table: {pwhy}",
                      "consequence": C.fallback("peak_table")}
    order, owhy = _ordered_targets(ne["targets"], target_order)
    if order is None:
        return None, {"status": "failed", "reason": owhy}
    alias = dict(PANEL.NAME) if alias is None else dict(alias)
    claimed = collections.defaultdict(set)
    for tf, t in ne["tf_target"]:
        claimed[t].add(alias.get(tf, tf))
    cols = ["Chr", "Start", "End", "Distance to TSS", "Gene Name"]
    try:
        pk = pd.read_csv(pk_path, sep="\t", usecols=cols)
    except ValueError as exc:
        return None, {"status": "skipped", "reason": f"{pk_path}: {exc}",
                      "consequence": C.fallback("peak_table")}
    items, no_peak = [], []
    for g in order:
        sub = pk[pk["Gene Name"] == g].copy()
        if not len(sub):
            no_peak.append(g)
            continue
        sub["ad"] = sub["Distance to TSS"].abs()
        for k, (_i, r) in enumerate(sub.sort_values("ad").head(max_peaks_per_gene).iterrows(), 1):
            items.append({"label": f"N_{g}_p{k}", "source": "network", "gene": g,
                          "chrom": str(r.Chr), "peak": [int(r.Start) - 1, int(r.End)],
                          "claimed_tfs": sorted(claimed[g])})
    obs, awhy = se_anno_observed(run_output)
    for it in items:
        fold, l2fc = obs.get(it["gene"], (None, None))
        it["obs_log2fc"], it["obs_atac_fold"] = l2fc, fold
    st = {"status": "ok", "reason": None, "net_source": "se_trn/se_cyjs.json",
          "peak_table": pk_path, "peak_table_source": pwhy, "target_order": target_order
          if isinstance(target_order, str) else list(target_order), "target_order_note": owhy,
          "n_targets": len(order), "n_items": len(items), "genes_with_no_peak": no_peak,
          "observed_note": awhy}
    if awhy:
        st["status"], st["reason"] = "partial", awhy
    return items, st


def panel_items(run_output, ranked=PANEL.RANKED_TSV, top_peaks_per_tf=8, max_tfs=12,
                genome_fa=HG38_FA, alias=None):
    """(items, status) -- stage 2a's mrag_axis.stat.panel.panel_items(), tagged source='panel'.

    Delegated, not reimplemented: panel.py is the module that was measured object-for-object
    against the published panel items (46 PRAD, 35 OC2) and against the deleted node's own
    ag_items.json (80/80, 96/96).  Only the 'source' tag is added, which agprep_D.py set when it
    appended these items to the network ones.
    """
    p = f"{run_output}/{ranked}"
    if not os.path.exists(p):
        return None, {"status": "skipped", "reason": f"no ranked consensus table: {p} missing",
                      "consequence": C.fallback("consensus_ranked")}
    counts: dict = collections.Counter()
    try:
        ag, tf_order = PANEL.ag_items(run_output, ranked=ranked,
                                      top_peaks_per_tf=top_peaks_per_tf, max_tfs=max_tfs,
                                      genome_fa=genome_fa, counts=counts)
        items = PANEL.panel_items(run_output, ranked=ranked, genome_fa=genome_fa,
                                  name_map=dict(PANEL.NAME) if alias is None else dict(alias),
                                  items=ag)
    except Exception as exc:
        return None, {"status": "skipped", "reason": f"{p}: {type(exc).__name__}: {exc}",
                      "consequence": C.fallback("consensus_ranked")}
    for it in items:
        it["source"] = "panel"
    return items, {"status": "ok", "reason": None, "ranked": p, "n_ag_items": len(ag),
                   "tf_order": tf_order, "n_items": len(items), "dropped_rows": dict(counts)}


def candidate_items(run_output, max_peaks_per_gene=3, target_order="network", alias=None,
                    ranked=PANEL.RANKED_TSV, top_peaks_per_tf=8, max_tfs=12, genome_fa=HG38_FA):
    """(network items + panel items, {network: status, panel: status}).

    The concatenation order is the contract: build() walks this list against ONE RNG stream, so
    every network item's draws sit before every panel item's, exactly as agprep_D.py appended them.
    """
    net, nst = network_items(run_output, max_peaks_per_gene=max_peaks_per_gene,
                             target_order=target_order, alias=alias)
    pan, pst = panel_items(run_output, ranked=ranked, top_peaks_per_tf=top_peaks_per_tf,
                           max_tfs=max_tfs, genome_fa=genome_fa, alias=alias)
    return (net or []) + (pan or []), {"network": nst, "panel": pst}


def build(run, max_peaks_per_gene=3, score_frac=0.85, max_motifs=40, seed=1,
          trn_order="network", target_order="network", trn_tfs=None, foreign=None,
          jaspar=JASPAR_MEME, genome_fa=HG38_FA, gtf=REFGENE_GTF, ranked=PANEL.RANKED_TSV,
          top_peaks_per_tf=8, max_tfs=12, items=None, agent_root=C.AGENT_ROOT):
    """The target-end edits for one run: {status, reason, window, trn, foreign, items, detail}.

    `run` is a run id, a run directory or a run's output directory -- mrag_axis.stat.config.run_output
    resolves all three.  Returns a status instead of raising on every missing input, because the
    node this feeds runs by default with no on/off switch.

    seed defaults to 1, which is agprep_D.py's default and NOT edits_enhancer.build's 0: the two
    ends were seeded differently by the research and each stored result set is keyed to its own
    stream, so the default has to differ too.
    """
    from pyfaidx import Fasta

    out = C.run_output(run, agent_root)
    G = C.corgi_channels()
    WINDOW, HALF = G["WINDOW"], G["HALF"]
    res = {"status": "ok", "reason": None, "window": WINDOW, "trn": [], "foreign": [], "items": []}

    trn, neg, no_pwm, tst = trn_and_foreign(out, trn_order=trn_order, trn_tfs=trn_tfs,
                                            foreign=foreign, jaspar=jaspar, agent_root=agent_root)
    if trn is None:
        return {**res, **{k: tst[k] for k in ("status", "reason")}, "detail": tst}
    names, _nwhy = C.trn_tf_names(out)
    alias, alwhy = homer_alias(names or trn, no_pwm or [], agent_root=agent_root)
    if items is not None:
        # The injection seam the suite needs: candidate_items() is the only thing it bypasses, so
        # a doctored item can prove each per-item drop rule fires and that a drop consumes no
        # draws.  mrag_axis.stat.panel.panel_items takes the same `items=` for the same reason.
        cand = list(items)
        cst = {k: {"status": "ok", "reason": None, "source": "caller"}
               for k in ("network", "panel")}
    else:
        cand, cst = candidate_items(out, max_peaks_per_gene=max_peaks_per_gene,
                                    target_order=target_order, alias=alias, ranked=ranked,
                                    top_peaks_per_tf=top_peaks_per_tf, max_tfs=max_tfs,
                                    genome_fa=genome_fa)
    if not cand:
        bad = [s for s in cst.values() if s and s["status"] != "ok"]
        return {**res, "status": bad[0]["status"] if bad else "skipped",
                "reason": "; ".join(s["reason"] for s in bad) or "no candidate items",
                "detail": {"run_output": out, "sources": cst, "factors": tst}}
    idx, gst = gene_index(gtf)
    if idx is None:
        return {**res, "status": gst["status"], "reason": gst["reason"],
                "detail": {"run_output": out, "sources": cst, "factors": tst, "gtf": gst}}

    pwms = load_pwms(set(trn) | set(neg), jaspar=jaspar)
    trn = [t for t in trn if t in pwms]
    neg = [t for t in neg if t in pwms]
    res["trn"], res["foreign"] = trn, neg
    fa = Fasta(genome_fa)
    rng = np.random.default_rng(seed)                 # ONE stream for the whole build
    skipped: dict = collections.Counter()
    dropped = []
    for it in cand:
        ps, pe = it["peak"]
        chrom = it["chrom"]
        centre = (ps + pe) // 2
        ws = centre - HALF
        try:
            clen = len(fa[chrom])
        except Exception:
            # A skip consumes NO draws, so the stream for every surviving item is unchanged and
            # the run still reports a section (research would have raised here).
            skipped["bounds"] += 1
            dropped.append((it["label"], f"{chrom} not in {genome_fa}"))
            continue
        if ws < 0 or ws + WINDOW > clen:
            skipped["bounds"] += 1
            dropped.append((it["label"], "524-kb window runs off the chromosome"))
            continue
        gm = gene_model(it["gene"], chrom, centre, index=idx)
        if gm is None:
            skipped["no_gene_model"] += 1
            dropped.append((it["label"], f"{it['gene']} has no refGene transcript on {chrom}"))
            continue
        if not tss_in_window(gm["tss"], ws, WINDOW):
            skipped["tss_outside_window"] += 1
            dropped.append((it["label"], "TSS outside [ws+1000, ws+W-1000)"))
            continue
        wt = str(fa[chrom][ws:ws + WINDOW].seq).upper()
        lo_i, hi_i = ps - ws, pe - ws
        clip = lambda a, b: [max(a, ws), min(b, ws + WINDOW)]
        # Window-relative zones a control edit must not touch: the peak, the promoter, every exon.
        excl = [(lo_i, hi_i),
                (gm["tss"] - PROM_EXCL - ws, gm["tss"] + PROM_EXCL - ws)] + \
               [(a - EXON_PAD - ws, b + EXON_PAD - ws) for a, b in gm["exons"]]
        windows = {"peak": [[ps, pe]], "whole": [[ws, ws + WINDOW]],
                   "tss500": [clip(gm["tss"] - 500, gm["tss"] + 500)],
                   "prom1k": [clip(gm["tss"] - 1000, gm["tss"] + 1000)],
                   "exons": [clip(a, b) for a, b in gm["exons"]
                             if b > ws and a < ws + WINDOW],
                   "body": [clip(*gm["body"])]}
        rec = {"label": it["label"], "source": it["source"], "gene": it["gene"],
               "strand": gm["strand"], "tss": gm["tss"], "chrom": chrom, "peak": [ps, pe],
               "ws": ws, "wt_md5": hashlib.md5(wt.encode()).hexdigest(),
               "obs_log2fc": it["obs_log2fc"], "obs_atac_fold": it["obs_atac_fold"],
               "claimed_tfs": it["claimed_tfs"], "windows": windows, "tfs": {}}
        for tf in trn + neg:
            hits = [(lo_i + p, L) for p, L in scan(wt[lo_i:hi_i], pwms[tf], score_frac)]
            if not hits:
                continue
            hits = sample_hits(hits, max_motifs, rng)
            ko_e, ct_e = [], []
            for pos, L in hits:
                sh = list(wt[pos:pos + L]); rng.shuffle(sh)
                ko_e.append([int(pos), "".join(sh)])
                # Bounded rejection: three exclusion zones can in principle leave no room, and a
                # `while True` would then hang the node.  Every rejected draw is consumed.
                for _ in range(10000):
                    q = int(rng.integers(EDGE_MARGIN, WINDOW - L - EDGE_MARGIN))
                    if not any(a - L < q <= b for a, b in excl):
                        break
                else:
                    skipped["control_fallback"] += 1
                sh = list(wt[q:q + L]); rng.shuffle(sh)
                ct_e.append([q, "".join(sh)])
            rec["tfs"][tf] = {"n_motifs": len(hits), "tf_class": "TRN" if tf in trn else "foreign",
                              "claimed": tf in it["claimed_tfs"], "ko": ko_e, "ctl": ct_e}
        res["items"].append(rec)

    pairs = [v for x in res["items"] for v in x["tfs"].values()]
    res["detail"] = {
        "run_output": out, "sources": cst, "factors": tst, "no_pwm": no_pwm, "gtf": gst,
        "alias": alias, "alias_note": alwhy, "seed": seed, "score_frac": score_frac,
        "max_motifs": max_motifs, "max_peaks_per_gene": max_peaks_per_gene, "jaspar": jaspar,
        "genome_fa": genome_fa,
        "n_candidates": len(cand),
        "n_network": sum(1 for x in res["items"] if x["source"] == "network"),
        "n_panel": sum(1 for x in res["items"] if x["source"] == "panel"),
        "skipped": dict(skipped), "dropped_items": dropped,
        "n_pairs": len(pairs), "n_motifs": sum(v["n_motifs"] for v in pairs),
        "n_claimed_pairs": sum(v["claimed"] for v in pairs),
        "n_trn_pairs": sum(v["tf_class"] == "TRN" for v in pairs),
        "n_foreign_pairs": sum(v["tf_class"] == "foreign" for v in pairs),
        # agrun.py bills one call per wild type plus one per knockout and control.
        "n_calls": len(res["items"]) + 2 * len(pairs)}
    # The run still reports a section; the status says which half of it is weaker and why.
    worst = [s for s in (tst, cst["network"], cst["panel"]) if s and s["status"] != "ok"]
    if worst:
        res["status"] = "failed" if any(s["status"] == "failed" for s in worst) else "partial"
        res["reason"] = "; ".join(str(s.get("reason")) for s in worst)
    return res


def published_form(res):
    """The four keys agprep_D*.py wrote, in its key order, for the byte/field comparison."""
    return {"window": res["window"], "trn": res["trn"], "foreign": res["foreign"],
            "items": res["items"]}
