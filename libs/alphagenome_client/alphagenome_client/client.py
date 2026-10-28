"""Low-level wrapper around alphagenome.models.dna_client.

Thread-safe singleton + simple retry policy. All higher-level helpers should
go through `get_client()` rather than instantiating their own DnaClient.
"""
from __future__ import annotations
import threading
import time
from functools import lru_cache
from typing import Iterable

from alphagenome.data import genome, ontology
from alphagenome.models import dna_client, dna_output

from .config import load_api_key

_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_client() -> dna_client.DnaClient:
    """Return the lazily-created DnaClient singleton (thread-safe)."""
    with _lock:
        return dna_client.create(api_key=load_api_key())


# ---- light convenience constructors -----------------------------------------

def interval(chrom: str, start: int, end: int) -> genome.Interval:
    return genome.Interval(chromosome=chrom, start=int(start), end=int(end))


def variant(chrom: str, pos: int, ref: str, alt: str) -> genome.Variant:
    return genome.Variant(
        chromosome=chrom, position=int(pos), reference_bases=ref, alternate_bases=alt
    )


def ontology_terms(terms: Iterable[str]) -> list[ontology.OntologyTerm]:
    return [ontology.from_curie(t) if isinstance(t, str) else t for t in terms]


# ---- output type helpers (so callers don't import dna_output directly) ------

OUTPUT_TYPES = {
    "ATAC": dna_output.OutputType.ATAC,
    "RNA_SEQ": dna_output.OutputType.RNA_SEQ,
    "CHIP_TF": dna_output.OutputType.CHIP_TF,
    "CHIP_HISTONE": dna_output.OutputType.CHIP_HISTONE,
    "DNASE": dna_output.OutputType.DNASE,
    "CAGE": dna_output.OutputType.CAGE,
    "PROCAP": dna_output.OutputType.PROCAP,
    "CONTACT_MAPS": dna_output.OutputType.CONTACT_MAPS,
    "SPLICE_SITES": dna_output.OutputType.SPLICE_SITES,
}


def resolve_output_types(names: Iterable[str]) -> list[dna_output.OutputType]:
    out = []
    for n in names:
        if n not in OUTPUT_TYPES:
            raise ValueError(f"Unknown OutputType '{n}'. Known: {sorted(OUTPUT_TYPES)}")
        out.append(OUTPUT_TYPES[n])
    return out


# ---- organism ---------------------------------------------------------------

ORGANISMS = {
    "human": dna_client.Organism.HOMO_SAPIENS,
    "hg38": dna_client.Organism.HOMO_SAPIENS,
    "mouse": dna_client.Organism.MUS_MUSCULUS,
    "mm10": dna_client.Organism.MUS_MUSCULUS,
}

DEFAULT_ORGANISM = "human"


def resolve_organism(name) -> dna_client.Organism:
    """Accept "human"/"hg38"/"mouse"/"mm10" or an Organism, and fail loudly.

    AlphaGenome mouse tracks are built on mm10 (GRCm38). Coordinates from any
    other mouse assembly are still valid numbers, so a mismatch would query the
    wrong locus without raising anything.
    """
    if isinstance(name, dna_client.Organism):
        return name
    key = str(name or DEFAULT_ORGANISM).strip().lower()
    if key not in ORGANISMS:
        raise ValueError(
            "Unknown organism %r. Known: %s" % (name, sorted(ORGANISMS)))
    return ORGANISMS[key]


# ---- retry wrapper ----------------------------------------------------------

def predict_variant_with_retry(
    interval_: genome.Interval,
    variant_: genome.Variant,
    requested_outputs: Iterable[str],
    tissue_curies: Iterable[str] | None = None,
    *,
    organism = DEFAULT_ORGANISM,
    max_retries: int = 3,
    backoff_s: float = 5.0,
):
    """predict_variant with simple exponential backoff."""
    client = get_client()
    outputs = resolve_output_types(requested_outputs)
    onto = ontology_terms(tissue_curies) if tissue_curies else None
    org = resolve_organism(organism)
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.predict_variant(
                interval=interval_,
                variant=variant_,
                organism=org,
                requested_outputs=outputs,
                ontology_terms=onto,
            )
        except Exception as e:  # noqa: BLE001 — re-raised after retries
            last_err = e
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_s * (2**attempt))
    raise RuntimeError("unreachable") from last_err