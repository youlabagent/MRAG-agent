"""mrag_axis.prep -- the RUN-DRIVEN enhancer end of the Section-13 axis-validation prep.

What the research pair ft/ag/agprep_A.py + ft/prad/agprep_A_prad.py did for one hardcoded RUN
each, this package does for an arbitrary MRAG run: it picks the super-enhancers and the
rank-stratified ordinary enhancers from that run's own ROSE table, scans the run's own TRN factors
plus the lineage-foreign null for JASPAR motifs, and emits the WT/knockout/control edit items that
the AlphaGenome and Corgi arms both replay.

Every per-run constant comes through stage 2a's mrag_axis.stat.config -- the ROSE table and its signal column
('*.bam' or '*_bigwig_signal'), the TRN factor list and its JASPAR filter, the foreign panel, the
SE->TF assignment and the 524-kb window geometry -- so nothing in this package names a run.

Exactness is the acceptance test, not a nicety: agprep_A_prad.py's docstring says it emits edits
"bit-identical to the Corgi run" by replaying seko2's random-number stream, and the stored
AlphaGenome/Corgi per-pair results are keyed to those exact edits.  test_prep_enhancer.py checks
this chain against both published artefacts, ft/ag/edits_A.json (OC2,
the super-enhancer run) and ft/prad/edits_A_prad.json (PRAD, the prostate-domain run),
item for item and field for field.

ONE input is not derivable from a run and is a declared parameter: the factor iteration order
(edits_enhancer.TRN_ORDER).  It fixes which draws of the single RNG stream go to which
(region, factor), and the two research scripts used different orders for the same rule -- PRAD the
run's se_cyjs TF-node order, OC2 that same set typed alphabetically.

Nothing here calls AlphaGenome, loads a Corgi checkpoint or touches a GPU: it writes edits.
Needs numpy, pandas and pyfaidx; tangermeme is NOT needed (motifs.read_meme replaces it, and the
suite checks the two readers agree on every motif in the file).
"""
from . import edits_enhancer, motifs, regions
from .edits_enhancer import (HG38_FA, TRN_ORDER, assigned_pairs, build, published_form,
                             trn_and_foreign)
from .motifs import JASPAR_MEME, load_pwms, read_meme, sample_hits, scan
from .regions import ROSE_FIXED, pick_regions, rose_table

__all__ = [
    "edits_enhancer", "motifs", "regions",
    # regions
    "ROSE_FIXED", "pick_regions", "rose_table",
    # motifs
    "JASPAR_MEME", "load_pwms", "read_meme", "sample_hits", "scan",
    # edits
    "HG38_FA", "TRN_ORDER", "assigned_pairs", "build", "published_form", "trn_and_foreign",
]
