"""Pick the enhancer-end regions from the run's OWN ROSE table.

Productionised from seko2.pick_regions / seko2_prad.pick_regions.  The research pair differs in
exactly one line -- the name of the ROSE signal column, '*.bam' for the OC2 run that had alignments
and '*_bigwig_signal' for the PRAD run that had a bigwig -- and that line is the reason two copies
of the function existed.  Here the column comes from mrag_axis.stat.config.rose_signal_col(), which reads the
table's own '#Ranked by <column>' header, so neither convention is hardcoded and a third one works.

The region set is deliberately NOT signal-matched (seko2's docstring): the separation of signal is
what defines a super-enhancer, so the ordinary enhancers are sampled evenly over their whole rank
spectrum instead, which lets the signal dependence of the effect be measured and separated from
super-enhancer status.

se_top10.tsv is not read here.  seko2 imports SE10 but pick_regions never uses it -- the top-n SE
list is the promoter end's input (mrag_axis.stat.config.se10 / mrag_axis.stat.config.peak_table) -- and taking the SE set
from se_top<top_n>.tsv instead would cap the enhancer end at that node's top_n parameter.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from mrag_axis.stat import config as C

# ROSE's own fixed fields; everything else on the header line is the signal column.
ROSE_FIXED = ("REGION_ID", "CHROM", "START", "STOP", "NUM_LOCI", "CONSTITUENT_SIZE",
              "enhancerRank", "isSuper")


def rose_table(run_output, agent_root=C.AGENT_ROOT):
    """(DataFrame, signal column, status).  Never raises: a run with no ROSE table reports
    status['status'] == 'skipped' and config.fallback('rose_all') as the consequence."""
    path, why = C.rose_all(run_output)
    if not path:
        return None, None, {"status": "skipped", "reason": f"no ROSE AllEnhancers table: {why}",
                            "consequence": C.fallback("rose_all")}
    sig, note = C.rose_signal_col(path)
    if not sig:
        return None, None, {"status": "skipped", "reason": f"{path}: {note}",
                            "consequence": C.fallback("rose_all")}
    # The research used skiprows=5; ROSE writes five '#' lines, but counting them keeps a table
    # whose provenance block grew (or shrank) readable instead of silently off by a row.
    skip = 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            skip += 1
    t = pd.read_csv(path, sep="\t", skiprows=skip)
    t.columns = [c.strip() for c in t.columns]
    if sig not in t.columns:
        return None, None, {"status": "skipped", "reason": f"signal column {sig!r} not in {path}",
                            "consequence": C.fallback("rose_all")}
    t = t.rename(columns={sig: "signal", "CHROM": "chrom", "START": "start", "STOP": "stop"})
    return t, sig, {"status": "ok", "table": path, "signal_col": sig,
                    "signal_col_note": note, "table_source": why, "skiprows": skip}


def pick_regions(run_output, n_se=40, n_ctl=40, seed=0, agent_root=C.AGENT_ROOT, table=None):
    """(DataFrame, status) -- the research region set, in the research ROW ORDER.

    Row order is load-bearing: edits_enhancer walks these rows in order against a single RNG
    stream, so the 40 super-enhancers by ascending enhancerRank must come before the ordinary
    enhancers, and the ordinary ones must stay in ascending-rank order.

    `seed` is accepted for signature parity with seko2.pick_regions, which also ignored it: the
    ordinary-enhancer sample is np.linspace over the rank order, which is deterministic.  Keeping
    the argument means the caller passes the run's one seed to one place.
    """
    t, sig, st = (table, None, {"status": "ok"}) if table is not None else \
        rose_table(run_output, agent_root=agent_root)
    if t is None:
        return None, st
    se = t[t.isSuper == 1].nsmallest(n_se, "enhancerRank").copy()
    rest = t[t.isSuper == 0].sort_values("enhancerRank").reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(rest) - 1, n_ctl).astype(int))   # stratified over rank
    ctl = rest.loc[idx].copy()
    se["cls"], ctl["cls"] = "SE", "ordinary"
    se["label"] = [f"SE{int(r)}" for r in se.enhancerRank]
    ctl["label"] = [f"ENH{int(r)}" for r in ctl.enhancerRank]
    reg = pd.concat([se, ctl], ignore_index=True)[
        ["label", "cls", "chrom", "start", "stop", "signal", "enhancerRank"]]
    st = dict(st)
    st.update({"n_se": int((reg.cls == "SE").sum()), "n_ordinary": int((reg.cls == "ordinary").sum()),
               "n_super_available": int((t.isSuper == 1).sum()), "n_rows": int(len(t))})
    return reg, st
