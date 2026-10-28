"""Section-13 axis validation: the SE-TF-target test the report's Section 13 shows (2026-10-18).

Three subpackages, each with its own acceptance test, in the order a run walks them:

  mrag_axis.prep   regions, the JASPAR motif scan and the edit items.  Reproduces the research
                   edits byte for byte on the four published domains (stage 2c-1), which matters
                   because the stored AlphaGenome and Corgi results are keyed to those exact items.
  mrag_axis.arm    the two models.  alphagenome calls predict_sequence on whole edited sequences;
                   corgi runs the installed checkpoint under a GPU lease (mrag_orchestrator.gpu_lease).
                   Neither raises: a dead API or no free card is a status and a reason.
  mrag_axis.stat   the statistic.  Pair-level Fisher against the lineage-foreign factors put through
                   the identical construction, leave-one-out null, the control-edit exclusion rule,
                   consensus intersected at row level.  `definition="published"` reproduces the
                   2026-10 artefacts; "unified" is what the report uses (SECTION13_CONTRACT.md).

Nothing here is imported at orchestrator start-up: the node imports what it needs when it runs.
"""
