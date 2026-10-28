"""Thin Python wrapper around AlphaGenome for the MRAG pipeline."""
from .helpers import counterfactual_at_peak
from .client import (
    get_client,
    predict_variant_with_retry,
    resolve_organism,
    ORGANISMS,
    DEFAULT_ORGANISM,
)
from .config import load_api_key, DEFAULT_TISSUE_PROSTATE

__all__ = [
    "counterfactual_at_peak",
    "get_client",
    "predict_variant_with_retry",
    "resolve_organism",
    "ORGANISMS",
    "DEFAULT_ORGANISM",
    "load_api_key",
    "DEFAULT_TISSUE_PROSTATE",
]
__version__ = "0.2.0"
