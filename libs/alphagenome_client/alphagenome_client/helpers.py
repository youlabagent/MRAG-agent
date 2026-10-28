"""High-level helpers used by the MRAG pipeline.

The most important entry point is `counterfactual_at_peak`, which simulates
disruption of a TF binding site at the centre of an ATAC peak and returns a
compact dict suitable for inclusion in an LLM reasoning chain.
"""
from __future__ import annotations
import json
import logging
from typing import Iterable

import numpy as np

from .cache import get as cache_get, hash_request, put as cache_put
from .client import interval, predict_variant_with_retry, variant
from .config import DEFAULT_TISSUE_PROSTATE

log = logging.getLogger(__name__)

# Default sequence window for variant prediction (16KB strikes a balance between
# nearby gene coverage and call latency).
_DEFAULT_WINDOW = 16384

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _scramble_motif(ref: str) -> str:
    """Return a heuristic 'disrupted' allele matched in length to *ref*.

    Strategy: reverse-complement. This breaks position-specific information for
    any palindromic-by-default motif while preserving GC content and length, so
    the resulting signal is interpretable as 'motif lost' rather than 'sequence
    composition changed'.
    """
    return ref.upper().translate(_COMPLEMENT)[::-1]


def _summarise_track(arr) -> dict:
    """Compact numeric summary of an output track (works for 1-D or 2-D)."""
    a = np.asarray(arr)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "max_abs": float(np.abs(a).max()),
        "sum": float(a.sum()),
    }


def counterfactual_at_peak(
    chrom: str,
    peak_start: int,
    peak_end: int,
    ref_at_centre: str,
    *,
    alt: str | None = None,
    window: int = _DEFAULT_WINDOW,
    requested_outputs: Iterable[str] = ("ATAC", "RNA_SEQ", "CHIP_TF"),
    tissue: str = DEFAULT_TISSUE_PROSTATE,
    organism: str = "human",
    use_cache: bool = True,
) -> dict:
    """Simulate the effect of disrupting a TF binding site at the centre of a peak.

    Args:
      chrom: chromosome name (e.g. "chr10").
      peak_start, peak_end: peak coordinates (0-based, half-open).
      ref_at_centre: reference allele at the peak centre (length 1+).
      alt: alternative allele. If None, a length-matched reverse-complement of
        *ref_at_centre* is used as a heuristic 'motif disruption'.
      window: total interval length around the variant. Must be one of the
        AlphaGenome SUPPORTED_SEQUENCE_LENGTHS (1KB, 16KB, 100KB, 500KB, 1MB).
      requested_outputs: which output modalities to request.
      tissue: ontology curie (default: prostate gland).
      organism: "human"/"hg38" or "mouse"/"mm10". Mouse tracks are built on
        mm10 (GRCm38), so mm39 coordinates must be lifted over before use;
        they would otherwise resolve to a valid but wrong locus.
      use_cache: read/write the local SQLite cache.

    Returns:
      Dict with REF/ALT summaries per modality plus delta signals. Designed to
      be small enough to fit comfortably in an LLM prompt.
    """
    centre = (int(peak_start) + int(peak_end)) // 2
    half = window // 2
    iv_start = centre - half
    iv_end = centre + half

    ref = ref_at_centre.upper()
    if alt is None:
        alt = _scramble_motif(ref)
    if alt == ref:
        # fall back to a single-base substitution if scramble is a no-op
        alt = "A" if ref[0] != "A" else "T"

    request = {
        "chrom": chrom,
        "pos": centre,
        "ref": ref,
        "alt": alt,
        "interval": [iv_start, iv_end],
        "outputs": sorted(requested_outputs),
        "tissue": tissue,
        "organism": str(organism).lower(),
    }
    key = hash_request(request)

    if use_cache:
        hit = cache_get(key)
        if hit is not None:
            log.info("cache hit %s", key[:12])
            return hit["summary"]

    iv = interval(chrom, iv_start, iv_end)
    var = variant(chrom, centre, ref, alt)
    response = predict_variant_with_retry(
        iv, var, requested_outputs=requested_outputs, tissue_curies=[tissue],
        organism=organism,
    )

    summary = {"request": request, "tracks": {}}
    for ot in requested_outputs:
        ref_arr = getattr(response.reference, ot.lower(), None)
        alt_arr = getattr(response.alternate, ot.lower(), None)
        if ref_arr is None or alt_arr is None:
            continue
        ref_values = ref_arr.values if hasattr(ref_arr, "values") else ref_arr
        alt_values = alt_arr.values if hasattr(alt_arr, "values") else alt_arr
        delta = np.asarray(alt_values) - np.asarray(ref_values)
        summary["tracks"][ot] = {
            "ref": _summarise_track(ref_values),
            "alt": _summarise_track(alt_values),
            "delta": _summarise_track(delta),
        }

    if use_cache:
        cache_put(key, request, summary)
    return summary
