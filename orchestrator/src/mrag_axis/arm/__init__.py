"""mrag_axis.arm -- the model arms of the Section-13 axis-validation node.  This package holds the
AlphaGenome arm.

Where the other stage-2 packages sit: s13 (stage 2a) is the statistic, which SCORES stored
per-pair results; mrag_axis.prep writes the edit items; mrag_axis.arm turns those items into the stored
per-pair results the statistic reads.  The split is the reason stage 2a can be tested without an
API key and a GPU, and it is why this package owns the only code that spends AlphaGenome quota.

alphagenome.run_layer(edits, out) is the whole arm: one AlphaGenome layer -- an enhancer-end or a
target-end edits JSON -- replayed into the resumable JSONL + '.tracks.json' sidecar pair that
s13.stats loads, plus the status record the report needs to say which arm is missing and why.
It is /BIO1/MRAG/corgi/ft/ag/agrun.py's request and window arithmetic, vendored (the production
alphagenome_client exposes only predict_variant_with_retry and its cache cannot key a
whole-sequence request -- see alphagenome.py's module docstring), with the three things a node
needs and a research runner did not have: a status record, retries that return instead of
raising, and a status of ok | partial | no signal | failed.

Nothing in this package raises for a missing input: a missing edits file, a missing hg38, an
unimportable SDK, an unobtainable client, a dead API, a signal-free answer and a wt_md5 mismatch
are each a status and a reason, per the model-arm failure policy.  No API key is ever
returned, logged or written to a file.

The arm is validated WITHOUT calling AlphaGenome: test_arm_alphagenome.py injects a stub client
and pins the request and the window sums against the published artefacts (ft/ag/results_A.jsonl
and ft/prad/ag_A_prad_tf.jsonl, with ft/ag/agrun.py imported as the reference implementation).
"""
from . import alphagenome

# The Corgi arm is the sibling module in this package.  It is imported defensively because the
# two arms run in DIFFERENT interpreters by design -- the AlphaGenome arm needs the SDK, which
# only libs/alphagenome_client/.venv has, and the Corgi arm needs torch, which only
# libs/corgi_client/.venv has -- so neither arm may make the package unimportable for the other.
try:
    from . import corgi
except Exception as _e:  # noqa: BLE001 -- recorded, not raised; the other arm still works
    corgi, corgi_import_error = None, "%s: %s" % (type(_e).__name__, _e)
else:
    corgi_import_error = None
from .alphagenome import (AG_PY, CHIP_TF, HG38_FA, MODALITY, ONT_PUBLISHED, OUT_NAMES,
                          SIDECAR_COLS, STATUSES, Fa, apply_edits, done_keys, has_signal,
                          job_seqs, layer_request, make_client, modalities_for, plan_jobs,
                          predict_with_retry, probe, rebuild_wt, resolve_outputs, run_layer,
                          run_layer_child, track_key, tracks_sidecar, window_sums)

__all__ = [
    "alphagenome", "corgi", "corgi_import_error",
    # the published request
    "AG_PY", "CHIP_TF", "HG38_FA", "MODALITY", "ONT_PUBLISHED", "OUT_NAMES", "SIDECAR_COLS",
    "STATUSES", "layer_request", "make_client", "modalities_for", "probe", "resolve_outputs",
    # sequences
    "Fa", "apply_edits", "job_seqs", "plan_jobs", "rebuild_wt",
    # read-out
    "has_signal", "track_key", "tracks_sidecar", "window_sums",
    # the layer
    "done_keys", "predict_with_retry", "run_layer", "run_layer_child",
]
