"""The Section-13 axis statistic: one parameterised implementation for all four domains.

Everything is read from the stored per-pair results (the AlphaGenome JSONL plus its
'.tracks.json' sidecar, the Corgi JSONL, the seko2 TSV and the edits JSONs).  No model is
called and no API is contacted -- stage 2a is scoring only.

`definition` selects which branch is scored:

  "unified"    the agreed definition stated in mrag_axis/stat/__init__.py: the pair is the unit, the null
               threshold comes from the row's own leave-one-out set, the target end is grouped
               by gene, the control-edit exclusion is computed, consensus is intersected at row
               level.  This is what a production node reports.
  "published"  the exact code path of the 2026-10 research scripts (axis_models_loo.py on the
               three OC2 domains, axis_v3.py --pol off --ar both on PRAD).  It exists ONLY to
               reproduce the published artefacts byte-for-byte; its numbers are not the node's.

The two differ in more than a threshold, which is why both code paths are kept:
  * target-end grouping -- published builds the null per promoter-peak item and keys the pass by
    gene, so a gene with k peaks gets k chances against k small nulls; unified groups by gene.
  * the enhancer-end exclusion -- published drops the three hand-listed OC2 pairs and nothing on
    PRAD; unified derives the set with ctl_in_readout_bin().
  * the effect formula -- published goes through the wild type (agstats.effect, i.e.
    log2((ko+1)/(wt+1)) - log2((ctl+1)/(wt+1))); unified cancels the wild type algebraically
    (log2((ko+1)/(ctl+1))).  The two agree to 4.4e-16 (stats/s13_check.py measured it on
    1434 pairs), which is far below any reported figure but above a floating-point tie, so each
    branch keeps the form its numbers were measured with.
"""
import json, math, os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# Research result roots.  Parameters everywhere below; these are only the current defaults.
AG_ROOT = "/BIO1/MRAG/corgi/ft/ag"
PRAD_ROOT = "/BIO1/MRAG/corgi/ft/prad"
SEKO2_TSV = "/BIO1/MRAG/corgi/ft/data/seko2_LNCaP_OE.tsv"

# Corgi window geometry, as corgi_run_prad.py sums it (mrag_axis.stat.config.corgi_channels() derives the
# same three numbers from corgi.constants; repeated here so stats.py imports nothing from Corgi).
BIN, FLANK, NBIN = 64, 65_536, 6_144
# A comparison group (an SE's or a gene's factors) is scored when it holds at least this many
# lineage-foreign rows.  The agreed definition uses 3; the 2026-10
# research scripts used 4, and definition='published' keeps 4 so its artefacts still reproduce.
MIN_FOREIGN = 3
MIN_FOREIGN_PUBLISHED = 4       # axis.py / axis_models*.py / axis_v*.py: len(frn) < 4 -> skip
EXCL_LIST = {("SE10", "CDX2"), ("SE21", "EGR1"), ("SE35", "FOXF1")}   # the OC2 hand list

# AlphaGenome track-key substrings, copied from /BIO1/MRAG/corgi/ft/ag/agstats.py so that a
# consolidated node does not import from the research directory.
TRACKS = {
    "dnase_LNCaP":      "dnase|EFO:0005726",
    "h3k4me3_LNCaP":    "EFO:0005726 Histone ChIP-seq H3K4me3",
    "h3k27ac_22Rv1":    "EFO:0002095 Histone ChIP-seq H3K27ac",
    "h3k27ac_VCaP":     "EFO:0007752 Histone ChIP-seq H3K27ac",
    "atac_PC3":         "atac|EFO:0002074",
    "dnase_PC3":        "dnase|EFO:0002074",
    "h3k27ac_PC3":      "EFO:0002074 Histone ChIP-seq H3K27ac",
    "h3k4me1_PC3":      "EFO:0002074 Histone ChIP-seq H3K4me1",
    "dnase_prostate":   "dnase|UBERON:0002367",
    "h3k27ac_prostate": "UBERON:0002367 Histone ChIP-seq H3K27ac",
}
RNA = {
    "rna_PC3":            "rna_seq|EFO:0002074",
    "rna_prostate_total": "rna_seq|UBERON:0002367 total",
    "rna_prostate_gtex":  "gtex",
    "cage_PC3":           "cage|hCAGE EFO:0002074",
    "cage_prostate":      "cage|hCAGE UBERON:0002367",
}

ENH_RD = ("h3k27ac", "dnase")
TGT_RD = ("rna", "prom", "dnase")
MODELS = {"corgi": ("corgi",), "alphagenome": ("ag",), "both": ("ag", "corgi")}
NAME = {"GR": "NR3C1", "GRE": "NR3C1", "FOX_EBOX": None, "FOXM1": None}
norm = lambda t: NAME.get(t, t)

# target-end read-out cells, exactly as each published script names them
TGT_OC2 = {"rna":   ((RNA["rna_prostate_total"], "exons"), ("corgi|rna_total", "exons")),
           "prom":  ((TRACKS["h3k4me3_LNCaP"], "prom1k"), ("corgi|h3k4me3", "prom1k")),
           "dnase": ((TRACKS["dnase_LNCaP"], "peak"), ("corgi|dnase", "peak"))}
TGT_PRAD = {"rna":   (("rna_seq|UBERON:0002367 total", "exons"), ("corgi|rna_total", "exons")),
            "prom":  (("UBERON:0002367 Histone ChIP-seq H3K4me3", "prom1k"),
                      ("corgi|h3k4me3", "prom1k")),
            "dnase": (("dnase|UBERON:0002367", "peak"), ("corgi|dnase", "peak"))}


def domains(ag_root=AG_ROOT, prad_root=PRAD_ROOT):
    """{domain: the stored per-pair result files it is scored from}.

    'enh_ag' is the A-layer AlphaGenome run the unified branch reads; 'enh_ag_pub' is the
    --chip-tf rerun axis_v3.py read instead (PRAD only).  Both exist on disk and they differ in
    their track set, not in their region sums, which is why the enhancer counts agree.
    """
    A, P = ag_root, prad_root
    oc2 = lambda tag, net: dict(
        branch="oc2", net=f"{A}/{net}", edits=f"{A}/edits_{tag}.json",
        ag=f"{A}/results_{tag}.jsonl", cg=f"{A}/corgi_{tag}.jsonl", tgt=TGT_OC2,
        enh_edits=f"{A}/edits_A.json", enh_ag=f"{A}/results_A.jsonl",
        enh_ag_pub=f"{A}/results_A.jsonl", enh_cg=None, enh_pub=f"{A}/candidates_all.tsv",
        seko2=SEKO2_TSV)
    return {
        "pc":   dict(oc2("D", "net_edges_pcq38.json"), axis_pub=f"{A}/axis_models_pc.json"),
        "nec":  dict(oc2("nec", "net_edges_necq38.json"), axis_pub=f"{A}/axis_models_nec.json"),
        "sclc": dict(oc2("sclc", "net_edges_sclcq38.json"), axis_pub=f"{A}/axis_models_sclc.json"),
        "prad": dict(branch="prad", net=f"{P}/net_edges_prad_pc.json",
                     edits=f"{P}/edits_D_prad.json", ag=f"{P}/ag_D_prad_tf.jsonl",
                     cg=f"{P}/corgi_D_prad.jsonl", tgt=TGT_PRAD,
                     enh_edits=f"{P}/edits_A_prad.json", enh_ag=f"{P}/ag_A_prad.jsonl",
                     enh_ag_pub=f"{P}/ag_A_prad_tf.jsonl", enh_cg=f"{P}/corgi_A_prad.jsonl",
                     enh_pub=None, seko2=None,
                     axis_pub=f"{P}/axis_prad_final_off.json"),
    }


def load(path):
    """One stored per-pair result file and its track metadata sidecar.

    A MISSING file is an EMPTY result, not an error: the two arms write their own JSONL pair and
    one of them can fail or be skipped, and the frame builders below load both unconditionally.
    An absent file therefore means "this model has no column", which is what the node reports as
    a withheld model set -- it must not take the arm that worked down with it (FileNotFoundError),
    and it must not be faked with an empty stand-in at the arm's own result path either, because
    the arm's own resume then reads the stand-in's '{}' sidecar as its track set.
    """
    if not path or not os.path.exists(path):
        return [], {}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    try:
        meta = json.load(open(path + ".tracks.json"))
    except FileNotFoundError:
        meta = {}
    return rows, meta


def effect(rows, meta, track_sub, window, scale="agstats", nb=None):
    """{(label, tf): effect of removing the factor's motifs} for one track and one read-out.

    Stranded assays contribute both strands as absolute values, so the minus strand (stored
    negative by AlphaGenome) does not cancel the plus strand.

      scale='agstats'  log2((ko+1)/(wt+1)) - log2((ctl+1)/(wt+1))  -- agstats.effect verbatim,
                       the form every published number was computed with
      scale='sum1'     log2((ko+1)/(ctl+1))                        -- the same quantity with the
                       wild type cancelled; differs from 'agstats' by <=4.4e-16
      scale='sum001'   window sums rescaled to bin means, pseudocount 0.01 (seko2 convention)
    """
    keys = [k for k in meta if track_sub in k]
    if not keys:
        return {}
    nch = sum(len(meta[k].get("channels", [1])) for k in keys)
    wt, ko, ct = {}, {}, {}
    for r in rows:
        tot, ok = 0.0, False
        for k in keys:
            d = r["sums"].get(k)
            if d is not None and window in d:
                tot += abs(d[window]); ok = True
        if not ok:
            continue
        if r["kind"] == "wt":
            wt[r["label"]] = tot
        elif r["kind"] == "ko":
            ko[(r["label"], r["tf"])] = tot
        else:
            ct[(r["label"], r["tf"])] = tot
    out = {}
    for key, kv in ko.items():
        if key[0] not in wt or key not in ct:
            continue
        cv = ct[key]
        if scale == "agstats":
            base = wt[key[0]]
            out[key] = math.log2((kv + 1) / (base + 1)) - math.log2((cv + 1) / (base + 1))
        elif scale == "sum1":
            out[key] = math.log2((kv + 1) / (cv + 1))
        else:
            m = nb.get(key[0], 0) * nch
            if m <= 0:
                continue
            out[key] = math.log2((kv / m + .01) / (cv / m + .01))
    return out


def nbins(edits_path, window, bin_bp=BIN, flank=FLANK, nbin=NBIN):
    """{label: number of summed 64-bp bins} for one read-out window, as corgi_run_prad.py sums it."""
    out = {}
    for it in json.load(open(edits_path))["items"]:
        o0 = it["ws"] + flank
        n = 0
        for a, b in it["windows"][window]:
            i0 = max((a - o0) // bin_bp, 0)
            i1 = min(-(-(b - o0) // bin_bp), nbin)
            if i1 > i0:
                n += i1 - i0
        out[it["label"]] = n
    return out


# The exclusion guards the CONTROL: a control edit that lands in a bin the read-out sums changes
# the control value, so the pair's effect is not a clean knockout-minus-control.  The grid is the
# COARSEST read-out's: AlphaGenome's histone and TF ChIP tracks are 128 bp over the whole
# 524,288-bp window (its DNase / RNA / CAGE are 1 bp, Corgi is 64 bp over the central
# 393,216 bp), so a 128-bp window-relative grid is the conservative choice -- and it is the only
# one that reproduces the research hand list {(SE10,CDX2), (SE21,EGR1), (SE35,FOXF1)} exactly.
# A 64-bp grid misses SE10/CDX2, whose control edit sits in the read-out's last 128-bp bin but
# past its last 64-bp bin (stats/s13core.py.bak64 is that earlier, wrong version).
CTL_BIN = 128


def ctl_in_readout_bin(edits_path, window, res=CTL_BIN):
    """Pairs whose paired control edit lands in a bin the read-out window actually sums."""
    bad = set()
    ed = json.load(open(edits_path))
    W = int(ed["window"])
    for it in ed["items"]:
        ws = it["ws"]
        ivs = it["windows"].get(window) or []
        if not ivs:
            continue
        summed = set()
        for a, b in ivs:
            i0 = max((a - ws) // res, 0)
            i1 = min(-(-(b - ws) // res), W // res)
            summed |= set(range(i0, i1))
        for tf, v in it["tfs"].items():
            for pos, bases in v["ctl"]:
                # the edit spans [pos, pos+len-1]: either end inside a summed bin contaminates it
                if (pos // res) in summed or ((pos + len(bases) - 1) // res) in summed:
                    bad.add((it["label"], tf))
                    break
    return bad


def enh_pairs(e, E):
    """The enhancer-end pair row for the passes E: the network pairs against the lineage-foreign
    ones, on the frame's own rows.

    Factored out of _axis_unified so the node can score the enhancer end ALONE -- the end the
    evidence rests on -- when the target end could not be built, through the identical arithmetic.
    """
    ecls = dict(zip(zip(e.label, e.tf), e.tf_class))
    net = [k for k in zip(e.label, e.tf) if ecls[k] != "foreign"]
    frn = [k for k in zip(e.label, e.tf) if ecls[k] == "foreign"]
    a = len([k for k in net if k in E])
    c = len([k for k in frn if k in E])
    o, p = fisher(a, len(net), c, len(frn))
    return dict(passed=a, n=len(net), fp=c, n_frn=len(frn), odds_ratio=o, p=p)


def enhancer_only(cfg, e, models=("ag", "corgi"), min_rd=1, ar=False):
    """{"enhancer": row} (+ "enhancer_plus_AR") for one model set, with NO target end.

    The target end is descriptive, so a target end that could not be built must
    not take the measured enhancer end with it.  Same selection and same arithmetic as
    _axis_unified's enhancer half; the triples need both ends and are not reported here.

    The AR row carries _axis_unified's own column check: the AR gate is an ALPHAGENOME read-out
    (22Rv1 / VCaP H3K27ac), so a frame built without that arm's enhancer rows -- the Corgi set on
    its own rows -- has no ag_ar22 / ag_arvc column, and asking beats() for one is a KeyError that
    used to cost every model set of the run (review 3, #1).  Such a set gets no AR row.
    """
    E = rowsets(e, "label", "label", ENH_RD, models, min_rd=min_rd)
    out = {"enhancer": enh_pairs(e, E)}
    if ar and cfg.get("branch") == "prad" and all(("ag_" + k) in getattr(e, "columns", [])
                                                  for k in ("ar22", "arvc")):
        gate = beats(e, "label", "label", "ag_ar22") & beats(e, "label", "label", "ag_arvc")
        out["enhancer_plus_AR"] = enh_pairs(e, E & gate)
    return out


def beats(frame, group, key, col, min_foreign=MIN_FOREIGN):
    """{(key, tf)} for rows strictly below every lineage-foreign row of their own group.

    The threshold comes from the row's own leave-one-out comparison set, so a foreign row can
    register as a false positive instead of being structurally unable to pass: with one global
    1/(1+len(frn)) a foreign row's best attainable p is 1/len(frn), which makes the
    false-positive column zero by construction.
    """
    out = set()
    for _, g in frame.groupby(group):
        frn = g[g.tf_class == "foreign"]
        if len(frn) < min_foreign:
            continue
        for _, r in g.iterrows():
            o = frn.drop(r.name) if r.tf_class == "foreign" else frn
            if (1 + (o[col].values <= r[col]).sum()) / (1 + len(o)) <= 1 / (1 + len(o)):
                out.add((r[key], r.tf))
    return out


def rowsets(frame, group, key, rds, models, min_rd=1, min_foreign=MIN_FOREIGN):
    """Row-level passes: a read-out counts when every requested model passes on it IN THE SAME
    ROW; the row passes when at least min_rd read-outs do.

    Intersecting at row level is the point.  Collapsing rows to their gene before intersecting
    lets two different promoter peaks of one gene supply the two models, which is a third,
    unintended cross-match -- and 'both models' would then not be the same read-out at all.
    """
    cnt = {}
    for rd in rds:
        ss = [beats(frame, group, key, f"{m}_{rd}", min_foreign) for m in models]
        for k in (set.intersection(*ss) if len(ss) > 1 else ss[0]):
            cnt[k] = cnt.get(k, 0) + 1
    return {k for k, n in cnt.items() if n >= min_rd}


def fisher(a_pass, a_n, c_pass, c_n):
    orr, p = fisher_exact([[a_pass, a_n - a_pass], [c_pass, c_n - c_pass]], alternative="greater")
    return (None if np.isinf(orr) else float(orr)), float(p)


# ------------------------------------------------------------------ enhancer end ---------------
def _enh_published(cfg):
    """The enhancer frame each published script actually read.

    OC2: candidates_all.tsv as written by the research run.  Recomputing its AlphaGenome columns
    from results_A.jsonl reproduces them only to ~4e-16, which is enough to flip an exact tie, so
    the published branch reads the file.  The index is deliberately NOT reset: axis_models_loo.py
    did not reset it either, and beats() drops by label.
    PRAD: rebuilt from edits_A_prad.json + the --chip-tf AlphaGenome rerun, carrying the two
    AR-positive H3K27ac read-outs the AR gate needs.  No pairs are excluded on PRAD.
    """
    if cfg["branch"] == "oc2":
        d = pd.read_csv(cfg["enh_pub"], sep="\t")
        d = d[[(l, t) not in EXCL_LIST for l, t in zip(d.label, d.tf)]]
        return d, sorted(EXCL_LIST)
    ed = json.load(open(cfg["enh_edits"]))
    rows = {(it["label"], tf): {"label": it["label"], "cls": it["cls"], "tf": tf,
                                "tf_class": v["tf_class"]}
            for it in ed["items"] for tf, v in it["tfs"].items()}
    ag, cg = load(cfg["enh_ag_pub"]), load(cfg["enh_cg"])
    for nm, sub in (("h3k27ac", TRACKS["h3k27ac_prostate"]), ("dnase", TRACKS["dnase_prostate"]),
                    ("ar22", TRACKS["h3k27ac_22Rv1"]), ("arvc", TRACKS["h3k27ac_VCaP"])):
        for k, v in effect(*ag, sub, "region", scale="agstats").items():
            if k in rows:
                rows[k]["ag_" + nm] = v
    for nm, sub in (("h3k27ac", "corgi|h3k27ac"), ("dnase", "corgi|dnase")):
        for k, v in effect(*cg, sub, "region", scale="agstats").items():
            if k in rows:
                rows[k]["corgi_" + nm] = v
    return pd.DataFrame(rows.values()).dropna(), []


def _enh_unified(cfg, excl="rule"):
    """The enhancer frame of the agreed definition, with the control-edit rule applied.

    OC2 Corgi can only be read in the seko2 convention: the A-layer Corgi run was never stored as
    window sums and seko2 keeps only log-ratios, so the wild-type level needed to rescale them is
    gone.  That is a stored-data limit, not a choice.
    """
    drop = ctl_in_readout_bin(cfg["enh_edits"], "region") if excl == "rule" else \
           set(EXCL_LIST) if excl is True else set()
    if cfg["branch"] == "oc2":
        d = pd.read_csv(cfg["seko2"], sep="\t")[
            ["label", "cls", "rank", "signal", "tf", "tf_class", "assigned", "n_motifs",
             "h3k27ac_eff", "dnase_eff"]].copy()
        rows, meta = load(cfg["enh_ag"])
        for nm, sub in (("h3k27ac", TRACKS["h3k27ac_22Rv1"]), ("dnase", TRACKS["dnase_LNCaP"])):
            e = effect(rows, meta, sub, "region", scale="sum1")
            d[f"ag_{nm}"] = [e.get((l, t), np.nan) for l, t in zip(d.label, d.tf)]
        d = d.dropna(subset=["ag_h3k27ac", "ag_dnase"]).rename(
            columns={"h3k27ac_eff": "corgi_h3k27ac", "dnase_eff": "corgi_dnase"})
    else:
        ed = json.load(open(cfg["enh_edits"]))
        rows = {(it["label"], tf): {"label": it["label"], "cls": it["cls"], "rank": it["rank"],
                                    "signal": it["signal"], "tf": tf, "tf_class": v["tf_class"],
                                    "assigned": v.get("assigned"), "n_motifs": v["n_motifs"]}
                for it in ed["items"] for tf, v in it["tfs"].items()}
        ag, cg = load(cfg["enh_ag"]), load(cfg["enh_cg"])
        nb = nbins(cfg["enh_edits"], "region")
        for nm, sub in (("h3k27ac", TRACKS["h3k27ac_prostate"]),
                        ("dnase", TRACKS["dnase_prostate"]),
                        ("ar22", TRACKS["h3k27ac_22Rv1"]), ("arvc", TRACKS["h3k27ac_VCaP"])):
            for k, v in effect(*ag, sub, "region", scale="sum1").items():
                if k in rows:
                    rows[k]["ag_" + nm] = v
        for nm, sub in (("h3k27ac", "corgi|h3k27ac"), ("dnase", "corgi|dnase")):
            for k, v in effect(*cg, sub, "region", scale="sum1", nb=nb).items():
                if k in rows:
                    rows[k]["corgi_" + nm] = v
        d = pd.DataFrame(rows.values()).dropna()
    d = d[[(l, t) not in drop for l, t in zip(d.label, d.tf)]]
    return d.reset_index(drop=True), sorted(drop)


def enh_frame(domain, definition="unified", excl="rule", cfg=None, **roots):
    """(enhancer frame, pairs dropped) for one domain under one definition."""
    cfg = cfg or domains(**roots)[domain]
    return _enh_published(cfg) if definition == "published" else _enh_unified(cfg, excl=excl)


# ------------------------------------------------------------------ target end -----------------
def _tgt_frame(cfg, definition, src="all"):
    """One row per (promoter-peak item, factor), carrying its gene.

    `src` keeps only edit items with that `source` ('network' / 'panel'); 'all' keeps both.  It is
    the knob the panel-inertness check uses -- the published numbers were computed with 'all'.
    """
    ed = json.load(open(cfg["edits"]))
    items = [it for it in ed["items"] if src == "all" or it.get("source") == src]
    rows = {(it["label"], tf): {"label": it["label"], "tf": tf, "gene": it["gene"],
                                "tf_class": v["tf_class"]}
            for it in items for tf, v in it["tfs"].items()}
    ag, cg = load(cfg["ag"]), load(cfg["cg"])
    sc = "agstats" if definition == "published" else "sum1"
    for rd, ((asub, awin), (csub, cwin)) in cfg["tgt"].items():
        nb = nbins(cfg["edits"], cwin)
        for k, v in effect(*ag, asub, awin, scale=sc).items():
            if k in rows:
                rows[k]["ag_" + rd] = v
        for k, v in effect(*cg, csub, cwin, scale=sc, nb=nb).items():
            if k in rows:
                rows[k]["corgi_" + rd] = v
    b = pd.DataFrame(rows.values()).dropna()
    return b if definition == "published" else b.reset_index(drop=True)


def tgt_frame(domain, definition="unified", src="all", cfg=None, **roots):
    return _tgt_frame(cfg or domains(**roots)[domain], definition, src=src)


# ------------------------------------------------------------------ triples --------------------
def _grid(cfg, e, b):
    """The SE-TF-target triples the network draws, and the identical grid built from the
    lineage-foreign factors, which is the false-positive count."""
    net = json.load(open(cfg["net"]))
    se_tf, tf_tg = set(map(tuple, net["se_tf"])), set(map(tuple, net["tf_target"]))
    SEm, TGm = set(e[e.cls == "SE"].label), set(b.gene)
    tri = sorted({(s, norm(t), g) for s, t in se_tf for t2, g in tf_tg
                  if norm(t) == norm(t2) and norm(t) is not None and s in SEm and g in TGm})
    SEuse, TGuse = sorted({x[0] for x in tri}), sorted({x[2] for x in tri})
    forTF = sorted(set(e[e.tf_class == "foreign"].tf) & set(b[b.tf_class == "foreign"].tf))
    ftri = [(s, t, g) for s in SEuse for t in forTF for g in TGuse]
    return tri, ftri, SEuse, TGuse, forTF


def triple_stats(cfg, e, b, E, B):
    """Descriptive triple counts.  No Fisher test is reported on triples in the agreed
    definition: 10 SE x 20 targets multiplies one enhancer-end pass into up to 20 triples, so the
    test would count the same evidence many times."""
    tri, ftri, _se, _tg, _f = _grid(cfg, e, b)
    nok = [x for x in tri if (x[0], x[1]) in E and (x[2], x[1]) in B]
    fok = [x for x in ftri if (x[0], x[1]) in E and (x[2], x[1]) in B]
    orr, p = fisher(len(nok), len(tri), len(fok), len(ftri))
    return dict(axes=len(nok), n_triples=len(tri), false=len(fok), n_foreign=len(ftri),
                odds_ratio=orr, p=p, SE=len({x[0] for x in nok}), TF=len({x[1] for x in nok}),
                target=len({x[2] for x in nok})), set(nok)


# ------------------------------------------------------------------ the two definitions --------
def _axis_published(cfg, src="all", min_rd=1):
    """axis_models_loo.py (OC2) / axis_v3.py --pol off --ar both (PRAD), reproduced.

    Key order and nesting are the published scripts', because axis_bytes() has to serialize to
    the same bytes as ft/ag/axis_models_<dom>.json and ft/prad/axis_prad_final_off.json.
    """
    e, _drop = _enh_published(cfg)
    b = _tgt_frame(cfg, "published", src=src)
    mf = MIN_FOREIGN_PUBLISHED
    ar = (beats(e, "label", "label", "ag_ar22", mf) & beats(e, "label", "label", "ag_arvc", mf)
          if cfg["branch"] == "prad" else None)
    tri, ftri, SEuse, TGuse, forTF = _grid(cfg, e, b)
    # The published target end builds the null per promoter-peak item (group='label') but reports
    # the pass by gene.  The consensus is still intersected at ROW level and only then keyed to
    # the gene: keying beats() by gene before intersecting would let two promoter peaks of one
    # gene supply the two models.  That is not hypothetical -- on nec with src='network' the
    # gene-keyed shortcut yields 13 false triples for 'both models' where the published file has
    # 12, so this line is what makes axis_models_nec.json reproduce.  axis_v3.py (PRAD) did
    # collapse to the gene before intersecting; on the PRAD data the two agree pair-for-pair, so
    # the row-level form is used for both ends of every domain.
    gene = dict(zip(zip(b.label, b.tf), b.gene))
    res = {}
    for key, mods in MODELS.items():
        E = rowsets(e, "label", "label", ENH_RD, mods, min_rd=min_rd, min_foreign=mf)
        if ar is not None:
            E = E & ar
        B = {(gene[k], k[1]) for k in rowsets(b, "label", "label", TGT_RD, mods, min_rd=min_rd,
                                              min_foreign=mf)}
        nok = [x for x in tri if (x[0], x[1]) in E and (x[2], x[1]) in B]
        fok = [x for x in ftri if (x[0], x[1]) in E and (x[2], x[1]) in B]
        orr, p = fisher(len(nok), len(tri), len(fok), len(ftri))
        res[key] = {"axes": len(nok), "n_triples": len(tri), "false": len(fok),
                    "n_foreign_triples": len(ftri), "fp_rate": len(fok) / len(ftri),
                    "or": orr, "p": p, "SE": len({x[0] for x in nok}),
                    "TF": len({x[1] for x in nok}), "target": len({x[2] for x in nok}),
                    "axes_list": [list(x) for x in nok]}
    if cfg["branch"] == "prad":
        res["grid"] = {"n_SE": len(SEuse), "n_targets": len(TGuse), "n_foreign_tf": len(forTF)}
    return res


def _axis_unified(cfg, src="all", min_rd=1, models=("ag", "corgi"), excl="rule"):
    """The agreed definition: the pair is the unit, the gene is the group at the target end.

    gnf is built as 'every (gene, factor) pair minus the network ones' rather than from the
    foreign rows directly: a factor can be foreign for one gene and a TRN member for another, and
    the comparison group is the gene.
    """
    e, dropped = _enh_frame_cached(cfg, excl)
    b = _tgt_frame(cfg, "unified", src=src)
    gene = dict(zip(zip(b.label, b.tf), b.gene))
    bcls = dict(zip(zip(b.label, b.tf), b.tf_class))
    gp = sorted({(gene[k], k[1]) for k in zip(b.label, b.tf)})
    gnt = sorted({(gene[k], k[1]) for k in zip(b.label, b.tf) if bcls[k] != "foreign"})
    gnf = [k for k in gp if k not in set(gnt)]
    out = {"dropped_pairs": [list(x) for x in dropped]}
    E = rowsets(e, "label", "label", ENH_RD, models, min_rd=min_rd)
    B = {(gene[k], k[1]) for k in rowsets(b, "gene", "label", TGT_RD, models, min_rd=min_rd)}
    out["enhancer"] = enh_pairs(e, E)
    a = len([k for k in gnt if k in B]); c = len([k for k in gnf if k in B])
    o, p = fisher(a, len(gnt), c, len(gnf))
    out["target"] = dict(passed=a, n=len(gnt), fp=c, n_frn=len(gnf), odds_ratio=o, p=p)
    tri, _st = triple_stats(cfg, e, b, E, B)
    out["graph_triples"] = {k: v for k, v in tri.items() if k not in ("odds_ratio", "p")}
    if cfg["branch"] == "prad" and all(("ag_" + k) in getattr(e, "columns", [])
                                       for k in ("ar22", "arvc")):
        # The AR gate stays a declared PRAD-only extra requirement, reported as its own row and
        # never folded into the headline, because no OC2 domain has an equivalent read-out.  It is
        # an ALPHAGENOME read-out, so a model set scored without that arm has no AR row at all --
        # asking beats() for a column the frame does not carry would be a KeyError.
        ar = beats(e, "label", "label", "ag_ar22") & beats(e, "label", "label", "ag_arvc")
        out["enhancer_plus_AR"] = enh_pairs(e, E & ar)
    return out


# The unified enhancer frame does not depend on the model set, and rebuilding it per model set
# triples the only slow step (the A-layer JSONL reads).  report.by_model() scores three model
# sets per domain, so it is cached by the files it is built from (see _enh_frame_cached).
_ENH_CACHE = {}


def _stamp(path):
    """(mtime_ns, size) of a result file, (0, -1) when it is absent -- part of the cache key, so
    an arm that filled its file in since the last scoring is not read from the cache."""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return (0, -1)
    return (st.st_mtime_ns, st.st_size)


def _enh_frame_cached(cfg, excl="rule"):
    # Keyed on every file the frame is built FROM, with its mtime and size: within one process the
    # same output directory can be scored twice (a re-run, or one model set per arm's own rows),
    # and a key of (edits, excl) alone would hand the second call the first call's columns.
    k = (str(cfg.get("branch")), str(excl)) + tuple(
        (str(cfg.get(x)),) + _stamp(cfg.get(x))
        for x in ("enh_edits", "enh_ag", "enh_cg", "enh_ag_pub", "enh_pub", "seko2"))
    if k not in _ENH_CACHE:
        d, drop = _enh_unified(cfg, excl=excl)
        _ENH_CACHE[k] = (d, drop)
    d, drop = _ENH_CACHE[k]
    return d.copy(), list(drop)


def axis(domain, definition="unified", src="all", min_rd=1, models=("ag", "corgi"),
         excl="rule", cfg=None, **roots):
    """The axis statistic for one domain.

    definition='unified'   -> the agreed definition's pair-level result for `models`
    definition='published' -> the 2026-10 research result, all three model sets, in the exact
                              shape of the published JSON (see axis_bytes)
    """
    cfg = cfg or domains(**roots)[domain]
    if definition == "published":
        return _axis_published(cfg, src=src, min_rd=min_rd)
    if definition == "unified":
        return _axis_unified(cfg, src=src, min_rd=min_rd, models=models, excl=excl)
    raise ValueError(f"definition must be 'unified' or 'published', not {definition!r}")


def axis_bytes(res):
    """The published serialization: json.dump(..., indent=1) and no trailing newline, which is
    what ft/ag/axis_models_<dom>.json and ft/prad/axis_prad_final_off.json were written with."""
    return json.dumps(res, indent=1).encode()


def pair_table(domain="prad", definition="published", cfg=None, **roots):
    """The enhancer end at the pair level -- the unit the enrichment actually rests on.

    This is pairtest.py's published table (its headline row is 'AlphaGenome + AR gate',
    302/987 vs 41/498, OR 4.91, p 8.0e-25).  It reads the A-layer AlphaGenome run without the
    --chip-tf rerun, which is the file that row was measured on.
    """
    cfg = cfg or domains(**roots)[domain]
    if definition == "published":
        # pairtest.py read ag_A_prad.jsonl, not the --chip-tf rerun axis_v3.py used.
        e, _ = _enh_published(dict(cfg, enh_ag_pub=cfg["enh_ag"]))
    else:
        e, _ = _enh_unified(cfg)
    mf = MIN_FOREIGN_PUBLISHED if definition == "published" else MIN_FOREIGN
    S = {f"{m}_{rd}": beats(e, "label", "label", f"{m}_{rd}", mf)
         for rd in ENH_RD for m in ("ag", "corgi")}
    ar = beats(e, "label", "label", "ag_ar22", mf) & beats(e, "label", "label", "ag_arvc", mf)
    cls = dict(zip(zip(e.label, e.tf), e.tf_class))
    nt = [k for k in zip(e.label, e.tf) if cls[k] != "foreign"]
    nf = [k for k in zip(e.label, e.tf) if cls[k] == "foreign"]
    cases = [("AlphaGenome (H3K27ac or DNase)", S["ag_h3k27ac"] | S["ag_dnase"]),
             ("AlphaGenome + AR gate", (S["ag_h3k27ac"] | S["ag_dnase"]) & ar),
             ("Corgi (H3K27ac or DNase)", S["corgi_h3k27ac"] | S["corgi_dnase"]),
             ("both models + AR gate",
              ((S["ag_h3k27ac"] & S["corgi_h3k27ac"])
               | (S["ag_dnase"] & S["corgi_dnase"])) & ar)]
    out = {}
    for nm, s in cases:
        a = len([k for k in nt if k in s]); c = len([k for k in nf if k in s])
        orr, p = fisher(a, len(nt), c, len(nf))
        out[nm] = {"pass": a, "n": len(nt), "fp": c, "n_frn": len(nf), "or": orr, "p": p}
    return out
