"""mrag_axis.stat -- the Section-13 axis validation, consolidated from the validated stage-1 prototypes.

THE AGREED DEFINITION (the contract; copied verbatim from stats/s13_node_definition.py, where each
decision was measured in s13_report.py rather than argued):

1. UNIT = the pair.  Two tests are reported per domain: (region, factor) at the enhancer end and
   (gene, factor) at the target end.  The triple is derived for the graph only -- an edge pair is
   drawn when both of its ends pass -- and carries no Fisher test, because 10 SE x 20 targets
   multiplies one enhancer-end pass into up to 20 triples.
2. NULL = within the comparison group the factor's effect must be strictly below every
   lineage-foreign factor's, with the threshold taken from the row's own leave-one-out set so a
   foreign row can register as a false positive.  Groups with < 3 foreign factors are not scored
   (the research scripts used 4, which definition='published' keeps).
3. GROUPING = the gene at the target end: a gene's promoter peaks are pooled into one comparison
   group and the gene is the unit.  Per-peak grouping gives a gene with k peaks k independent
   chances against k small nulls.
4. CONSENSUS = 'both models' means the same read-out passes under both models in the same row.
   It is NOT the intersection of the two single-model sets and must never be reported as one.
5. READ-OUTS = a fixed declared set, min_rd = 1, with the identical selection applied to the
   lineage-foreign column -- which is what prices the multiplicity in.  The Fisher p is then
   Holm-corrected over the headline tests the run reports.
6. CORGI EFFECT = log2((knockout + 1) / (control + 1)) on the model's own window sums, one code
   path at both ends for every domain.
7. CONTROL-EDIT EXCLUSION = a rule, not a list: drop a pair when any of its paired control edits
   lands in a bin that the read-out window sums.
8. The AR gate stays a PRAD-only, declared extra requirement; it is reported as a separate row,
   never folded into the headline, because no OC2 domain has an equivalent.

definition="published" EXISTS ONLY TO REPRODUCE THE 2026-10 ARTEFACTS.  It is the research code
path of axis_models_loo.py (OC2) and axis_v3.py --pol off --ar both (PRAD) -- per-peak grouping at
the target end, the three hand-listed OC2 exclusions, the wild-type-relative effect formula -- and
it is kept so ft/ag/axis_models_<dom>.json and ft/prad/axis_prad_final_off.json can be regenerated
byte-for-byte and so any change here is provably not a change to the published figures.  It is not
the definition a node should report.  Everything a production node prints comes from
definition="unified", i.e. report.by_model().

Nothing in this package calls AlphaGenome, loads a Corgi checkpoint, or touches a GPU: it scores
stored per-pair results.  Stage 2b/2c supply the runtime that produces those results.
"""
from . import config, panel, report, stats
from .config import (GATES, corgi_channels, excl_pairs, fallback, jaspar_names, lineage,
                     name_map, neg_tfs, net_edges, peak_table, rose_all, rose_signal_col,
                     run_output, se10, split_pwm, tf_cols, trn_tf_names, ag_biosamples,
                     ag_outputs, expr_context)
from .panel import ag_items, panel_items, published_panel_items, diff_panel
from .report import by_model, holm, table
from .stats import (axis, axis_bytes, beats, ctl_in_readout_bin, domains, effect, enh_frame,
                    fisher, load, pair_table, rowsets, tgt_frame, triple_stats)

__all__ = [
    "config", "panel", "report", "stats",
    # config
    "GATES", "corgi_channels", "excl_pairs", "fallback", "jaspar_names", "lineage", "name_map",
    "neg_tfs", "net_edges", "peak_table", "rose_all", "rose_signal_col", "run_output", "se10",
    "split_pwm", "tf_cols", "trn_tf_names", "ag_biosamples", "ag_outputs", "expr_context",
    # panel
    "ag_items", "panel_items", "published_panel_items", "diff_panel",
    # statistic
    "axis", "axis_bytes", "beats", "ctl_in_readout_bin", "domains", "effect", "enh_frame",
    "fisher", "load", "pair_table", "rowsets", "tgt_frame", "triple_stats",
    # report
    "by_model", "holm", "table",
]
