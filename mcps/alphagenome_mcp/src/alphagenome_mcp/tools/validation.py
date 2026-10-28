"""Validation tools — counterfactual prediction + Top-N TF batch validation.

`batch_validate_tfs` is the MRAG-specific entry point. It takes a list of
TF + top-peak records (typically built from mcp_homer output + mcp_diffbind
FC table), runs AlphaGenome counterfactual at each peak centre, and -- when
an `observed_log2fc` is supplied per item -- annotates whether the predicted
ATAC delta direction matches the observed direction.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# alphagenome_client lives in the same venv (editable install).
from alphagenome_client import counterfactual_at_peak as _cf_at_peak
from alphagenome_client.cache import get as _cache_get, hash_request as _hash_request
from alphagenome_client.config import (
    DEFAULT_TISSUE_PROSTATE,
    DEFAULT_TISSUE_ATAC,
    DEFAULT_TISSUE_RNA,
    CELL_LINES,
    MODALITY_COVERAGE,
    PROSTATE_CELL_LINES,
    PROSTATE_MODALITY_COVERAGE,
)
from alphagenome_client.helpers import _scramble_motif as _heuristic_scramble  # noqa: F401


def _peek_cache(
    chrom: str, peak_start: int, peak_end: int,
    ref_at_centre: str, alt,
    window: int, requested_outputs, tissue: str,
) -> bool:
    """True if a counterfactual call with these args would hit the cache.

    Mirrors request canonicalisation in alphagenome_client.helpers so we can
    report ``cache_hit`` accurately regardless of one-time SDK import cost.
    """
    centre = (int(peak_start) + int(peak_end)) // 2
    half = window // 2
    iv_start, iv_end = centre - half, centre + half
    ref = ref_at_centre.upper()
    if alt is None:
        alt = _heuristic_scramble(ref)
    if alt == ref:
        alt = "A" if ref[0] != "A" else "T"
    request = {
        "chrom": chrom, "pos": centre, "ref": ref, "alt": alt,
        "interval": [iv_start, iv_end],
        "outputs": sorted(requested_outputs),
        "tissue": tissue,
    }
    return _cache_get(_hash_request(request)) is not None


def _resolve_tissue(cell_line: Optional[str], explicit_tissue: Optional[str]) -> str:
    """Resolve a tissue ontology curie.

    Precedence:
      1. explicit_tissue (raw curie like "EFO:0002074" or "UBERON:0002367")
      2. cell_line key in PROSTATE_CELL_LINES
      3. PC-3 (DEFAULT_TISSUE_ATAC) if any output requires ATAC, else
         prostate_gland (DEFAULT_TISSUE_RNA)
    """
    if explicit_tissue:
        return explicit_tissue
    if cell_line:
        if cell_line in CELL_LINES:
            return CELL_LINES[cell_line]
        # Allow case-insensitive lookup
        for k, v in CELL_LINES.items():
            if k.lower() == cell_line.lower():
                return v
        raise ValueError(
            f"unknown cell_line '{cell_line}'. "
            f"Known ({len(CELL_LINES)}): {sorted(CELL_LINES)}"
        )
    return DEFAULT_TISSUE_ATAC  # PC-3 is safest default (only one with ATAC)


def _coverage_for(cell_line: Optional[str], explicit_tissue: Optional[str]):
    """Return the AlphaGenome MODALITY_COVERAGE dict for the resolved
    biosample, or None if coverage is unknown (e.g. a raw curie with no
    matching cell-line key). None => do not filter (trust the caller)."""
    if cell_line:
        if cell_line in MODALITY_COVERAGE:
            return MODALITY_COVERAGE[cell_line]
        for k, v in MODALITY_COVERAGE.items():
            if k.lower() == cell_line.lower():
                return v
    if explicit_tissue:
        # reverse-resolve a curie back to a cell-line key if possible
        for k, v in CELL_LINES.items():
            if v == explicit_tissue and k in MODALITY_COVERAGE:
                return MODALITY_COVERAGE[k]
    return None


def _filter_outputs(requested_outputs, coverage):
    """Drop requested outputs the biosample does not support.

    Returns (kept, dropped). When coverage is unknown, keeps everything.
    Never returns an empty ``kept`` — if none of the requested modalities
    are supported, the original list is returned so the caller surfaces a
    real AlphaGenome error rather than a silent no-op."""
    if not coverage:
        return list(requested_outputs), []
    kept, dropped = [], []
    for o in requested_outputs:
        if coverage.get(o, False):
            kept.append(o)
        else:
            dropped.append(o)
    if not kept:
        return list(requested_outputs), []
    return kept, dropped


def _sign(x: float) -> int:
    if x > 0:  return  1
    if x < 0:  return -1
    return 0

# ETS family TF gene symbols (capitalised; matched case-insensitively).
# Listed because their motifs (GGA-core) are palindrome-prone with respect
# to reverse-complement scramble -- empirically, RC fails to disrupt ETS
# binding (~38% obs_dir_match), while shuffle recovers to ~54-58%. Other
# families (bZIP, Zf) work better with the default RC scramble.
ETS_FAMILY_TFS = {
    "ETS1","ETS2","ERG","FLI1","ELK1","ELK3","ELK4",
    "ETV1","ETV2","ETV3","ETV4","ETV5","ETV6","ETV7",
    "ELF1","ELF2","ELF3","ELF4","ELF5",
    "EHF","ESE1","ESE2","ESE3",
    "GABPA","GABPB1","SPI1","SPIB","PU.1",
    "FEV","ERF","ERFL","EWSR1",
}


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def _alt_rc(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]

def _alt_n_mask(seq: str) -> str:
    return "N" * len(seq)

def _alt_shuffle(seq: str) -> str:
    """Deterministic shuffle: same input -> same output (cache-stable)."""
    import hashlib, random as _r
    s = seq.upper()
    seed = int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    rng = _r.Random(seed)
    chars = list(s)
    rng.shuffle(chars)
    out = "".join(chars)
    # if shuffle happens to be identical (rare for short seqs), force a swap
    if out == s and len(s) > 1:
        out = s[1] + s[0] + s[2:]
    return out

def _alt_random(seq: str) -> str:
    """Deterministic random ACGT of same length."""
    import hashlib, random as _r
    seed = int(hashlib.md5(seq.upper().encode()).hexdigest()[:8], 16)
    rng = _r.Random(seed)
    return "".join(rng.choices("ACGT", k=len(seq)))

_ALT_STRATEGIES = {
    "rc":      _alt_rc,
    "shuffle": _alt_shuffle,
    "n_mask":  _alt_n_mask,
    "random":  _alt_random,
}


def _resolve_alt_strategy_for_tf(tf: str, default: str = "rc") -> str:
    """auto-mode TF-family-aware strategy.

    ETS family (palindrome-prone) -> shuffle.
    All others -> the supplied default (typically "rc").
    """
    if tf and tf.upper() in ETS_FAMILY_TFS:
        return "shuffle"
    return default


def _compute_alt(ref: str, strategy: str) -> str:
    fn = _ALT_STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"unknown alt_strategy '{strategy}', "
            f"expected one of {sorted(_ALT_STRATEGIES)} or 'auto' (batch only)"
        )
    return fn(ref)




def register_validation_tools(mcp: FastMCP):

    @mcp.tool()
    def counterfactual_at_peak(
        chrom: str,
        peak_start: int,
        peak_end: int,
        ref_at_centre: str,
        alt: Optional[str] = None,
        alt_strategy: str = "rc",
        window: int = 16384,
        requested_outputs: Optional[List[str]] = None,
        cell_line: Optional[str] = "PC-3",
        tissue: Optional[str] = None,
        organism: str = "human",
        use_cache: bool = True,
    ) -> str:
        """Predict ATAC/RNA/CHIP_TF effect of disrupting a TF binding site.

        Strategy: take the reference allele at the peak centre, replace it
        with `alt` (default = reverse-complement = motif disruption), and
        compare AlphaGenome predictions for ref vs alt sequences. Returns
        compact per-track summaries (mean / max_abs / sum of the delta
        signal across the requested window).

        Args:
            chrom:           Chromosome (e.g. "chr11").
            peak_start, peak_end:
                             0-based half-open peak coordinates. Centre =
                             midpoint.
            ref_at_centre:   Reference allele at the centre. Length >= 1.
            alt:             Alternate allele. None -> reverse-complement of
                             ref_at_centre (preserves length + GC, drops
                             motif specificity).
            window:          AlphaGenome sequence window. Must be one of
                             1024, 16384, 100k, 500k, 1M.
            requested_outputs:
                             Subset of {ATAC, RNA_SEQ, CHIP_TF, CHIP_HISTONE,
                             DNASE, CAGE, PROCAP, CONTACT_MAPS, SPLICE_SITES}.
                             Default ["ATAC","RNA_SEQ","CHIP_TF"].
            cell_line:       Convenience key. "PC-3" (default), "LNCAP",
                             "22Rv1", "VCaP", "LNCaP_FGC", "prostate_gland",
                             "prostate_epi". Maps to the appropriate
                             EFO/UBERON ontology curie. Use 'PC-3' if you
                             need ATAC (only prostate cell line that has it).
            tissue:          Raw ontology curie. Overrides cell_line when
                             provided.
            use_cache:       Use the local SQLite cache (default True).

        Returns:
            JSON dict with the cached/computed summary plus a tiny header
            describing the resolved tissue and whether the result came from
            cache.
        """
        if requested_outputs is None:
            requested_outputs = ["ATAC", "RNA_SEQ", "CHIP_TF"]

        tissue_curie = _resolve_tissue(cell_line, tissue)
        _cov = _coverage_for(cell_line, tissue)
        requested_outputs, _dropped_outputs = _filter_outputs(requested_outputs, _cov)
        # Resolve alt: explicit alt wins; otherwise derive from alt_strategy.
        if alt is None:
            alt = _compute_alt(ref_at_centre, alt_strategy)
        cache_hit = use_cache and _peek_cache(
            chrom, int(peak_start), int(peak_end), ref_at_centre, alt,
            int(window), requested_outputs, tissue_curie,
        )
        t0 = time.time()
        summary = _cf_at_peak(
            chrom=chrom,
            peak_start=int(peak_start),
            peak_end=int(peak_end),
            ref_at_centre=ref_at_centre,
            alt=alt,
            window=int(window),
            requested_outputs=tuple(requested_outputs),
            tissue=tissue_curie,
            organism=organism,
            use_cache=use_cache,
        )
        elapsed = round(time.time() - t0, 3)

        return json.dumps(
            {
                "status":         "ok",
                "elapsed_s":      elapsed,
                "cache_hit":      bool(cache_hit),
                "cell_line":      cell_line,
                "tissue":         tissue_curie,
                "outputs_used":   requested_outputs,
                "outputs_dropped": _dropped_outputs,
                "summary":        summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def batch_validate_tfs(
        items: List[Dict[str, Any]],
        cell_line: str = "PC-3",
        tissue: Optional[str] = None,
        requested_outputs: Optional[List[str]] = None,
        window: int = 16384,
        use_cache: bool = True,
        alt_strategy: str = "auto",
        organism: str = "human",
    ) -> str:
        """Top-N TF validation: per TF, counterfactual at top peaks.

        This is the MRAG Step 8 entry point. It takes a list of items, one
        per TF×peak, runs AlphaGenome on each, and annotates direction-of-
        effect agreement with observed log2FC when supplied.

        Each item must have:
            tf:           TF gene symbol (e.g. "AR").
            chrom:        Chromosome.
            peak_start:   0-based start.
            peak_end:     0-based end (half-open).
            ref:          Reference allele at the centre.
        Optional fields:
            alt:                Alternate allele (default = reverse-complement).
            observed_log2fc:    Observed RNA log2FC for the TF (or target gene)
                                so we can compute obs_dir_match for RNA_SEQ.
            observed_atac_fc:   Observed DiffBind FC for the peak so we can
                                compute obs_dir_match for ATAC.
            label:              Free-form annotation echoed in output (e.g.
                                "rank=1 peak, target=KLK3").

        Direction-of-effect rules:
            obs_dir_match.atac = sign(ATAC.delta.sum) == sign(observed_atac_fc)
                                 with the convention that motif disruption
                                 should *decrease* accessibility, so for a
                                 TF that increases accessibility we expect
                                 ATAC.delta < 0 (i.e. sign opposite to
                                 observed_atac_fc).
            obs_dir_match.rna  = same but with observed_log2fc and
                                 RNA_SEQ.delta.

        Args:
            items:               List of dicts as described above.
            cell_line:           Default "PC-3" (only prostate cell line
                                 with ATAC). Switch to "prostate_gland" for
                                 RNA-only studies.
            tissue:              Raw ontology curie override.
            requested_outputs:   Default ["ATAC","RNA_SEQ","CHIP_TF"].
            window:              AlphaGenome window size (default 16384).
            use_cache:           Read/write local SQLite cache.

        Returns:
            JSON with header (cell_line / outputs / counts) and `rows`: per-
            item record carrying delta summaries, obs_dir_match (if observed
            FC supplied), elapsed time, and any error message.
            Plus `by_tf`: aggregated stats (n_peaks, n_obs_match, mean
            ATAC.delta.sum) per TF.
        """
        if requested_outputs is None:
            requested_outputs = ["ATAC", "RNA_SEQ", "CHIP_TF"]
        tissue_curie = _resolve_tissue(cell_line, tissue)
        _cov = _coverage_for(cell_line, tissue)
        requested_outputs, _dropped_outputs = _filter_outputs(requested_outputs, _cov)

        # Accessibility proxy: if the biosample carries no ATAC track but has
        # DNASE (e.g. kidney lines Caki2/G401), use DNASE in the ATAC slot so
        # S13/S14 still get a tissue-matched accessibility signal instead of a
        # wrong-tissue ATAC. access_key names which track fills the ATAC column.
        access_key = "ATAC"
        if _cov is not None and not _cov.get("ATAC", False) and _cov.get("DNASE", False):
            access_key = "DNASE"
            if "DNASE" not in requested_outputs:
                requested_outputs = list(requested_outputs) + ["DNASE"]

        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        t0 = time.time()

        for idx, item in enumerate(items):
            try:
                tf      = item.get("tf") or f"item_{idx}"
                chrom   = item["chrom"]
                pstart  = int(item["peak_start"])
                pend    = int(item["peak_end"])
                ref     = item["ref"]
                alt     = item.get("alt")
                label   = item.get("label")
                obs_l2fc = item.get("observed_log2fc")
                obs_atac = item.get("observed_atac_fc")

                # Resolve alt: explicit per-item alt wins; otherwise determine
                # strategy. "auto" = TF-family aware (ETS -> shuffle, else rc).
                if alt is None:
                    item_strat = item.get("alt_strategy")
                    if item_strat is None:
                        item_strat = (_resolve_alt_strategy_for_tf(tf, "rc")
                                      if alt_strategy == "auto" else alt_strategy)
                    alt = _compute_alt(ref, item_strat)

                t1 = time.time()
                summary = _cf_at_peak(
                    chrom=chrom,
                    peak_start=pstart,
                    peak_end=pend,
                    ref_at_centre=ref,
                    alt=alt,
                    window=int(window),
                    requested_outputs=tuple(requested_outputs),
                    tissue=tissue_curie,
                    organism=organism,
                    use_cache=use_cache,
                )
                dt = round(time.time() - t1, 3)

                tracks = summary.get("tracks", {})
                # A curie can be VALID for the organism and still carry no
                # tracks at all (mouse UBERON:0002367 returns ATAC/RNA_SEQ/
                # CHIP_TF each with n=0). Those empties flowed straight through:
                # delta.sum came back None, obs_dir_match became None, the row
                # was filtered out downstream and Section 13 rendered an empty
                # table under status "ok" (2026-09-23). An empty response is a
                # failed lookup, so say so.
                if tracks and all(
                        ((tracks.get(_m) or {}).get("ref") or {}).get("n", 0) == 0
                        for _m in tracks):
                    raise ValueError(
                        "AlphaGenome returned no data for biosample %r and "
                        "organism %r: every requested modality (%s) came back "
                        "with 0 positions. The ontology is accepted but carries "
                        "no track for this organism -- pick one that does."
                        % (tissue_curie, organism, ", ".join(sorted(tracks))))
                # extract scalar features (delta.sum is the most-reported)
                def _delta(name, field):
                    return tracks.get(name, {}).get("delta", {}).get(field)

                atac_d_sum = _delta(access_key, "sum")
                rna_d_sum  = _delta("RNA_SEQ", "sum")
                chip_d_sum = _delta("CHIP_TF", "sum")

                # Direction-of-effect agreement.
                # Motif disruption is expected to *decrease* the signal where
                # the TF normally binds. So if observed_atac_fc > 0
                # (i.e. peak increased in treatment vs ctrl), we expect
                # AG.ATAC.delta.sum < 0 (disrupting the TF that drove the
                # gain should erase it). => match when signs are OPPOSITE.
                obs_match = {}
                if atac_d_sum is not None and obs_atac is not None:
                    obs_match["atac"] = (_sign(atac_d_sum) == -_sign(obs_atac)) \
                                        if (atac_d_sum != 0 and obs_atac != 0) else None
                if rna_d_sum is not None and obs_l2fc is not None:
                    obs_match["rna"]  = (_sign(rna_d_sum)  == -_sign(obs_l2fc)) \
                                        if (rna_d_sum != 0 and obs_l2fc != 0) else None

                rows.append({
                    "idx":           idx,
                    "tf":            tf,
                    "label":         label,
                    "chrom":         chrom,
                    "peak_start":    pstart,
                    "peak_end":      pend,
                    "ref":           ref,
                    "alt_used":      summary.get("request", {}).get("alt"),
                    "ATAC_delta_sum": atac_d_sum,
                    "RNA_delta_sum":  rna_d_sum,
                    "CHIP_delta_sum": chip_d_sum,
                    "ATAC_delta_max_abs": tracks.get(access_key, {}).get("delta", {}).get("max_abs"),
                    "access_modality":     access_key,
                    "RNA_delta_max_abs":  tracks.get("RNA_SEQ",{}).get("delta", {}).get("max_abs"),
                    "CHIP_delta_max_abs": tracks.get("CHIP_TF",{}).get("delta", {}).get("max_abs"),
                    "obs_dir_match": obs_match if obs_match else None,
                    "elapsed_s":     dt,
                })
            except Exception as e:
                errors.append(f"item {idx}: {e!r}")
                rows.append({
                    "idx":     idx,
                    "tf":      item.get("tf"),
                    "error":   str(e),
                })

        # Aggregate per-TF stats
        by_tf: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if r.get("error"):
                continue
            tf = r["tf"]
            d = by_tf.setdefault(tf, {
                "n_peaks": 0,
                "n_atac_match": 0,  "n_atac_check": 0,
                "n_rna_match":  0,  "n_rna_check":  0,
                "mean_ATAC_delta_sum": [],
            })
            d["n_peaks"] += 1
            m = r.get("obs_dir_match") or {}
            if m.get("atac") is True:  d["n_atac_match"] += 1
            if m.get("atac") is not None: d["n_atac_check"] += 1
            if m.get("rna")  is True:  d["n_rna_match"]  += 1
            if m.get("rna")  is not None: d["n_rna_check"]  += 1
            if r.get("ATAC_delta_sum") is not None:
                d["mean_ATAC_delta_sum"].append(r["ATAC_delta_sum"])
        for tf, d in by_tf.items():
            v = d.pop("mean_ATAC_delta_sum")
            d["mean_ATAC_delta_sum"] = round(sum(v) / len(v), 4) if v else None

        _n_ok = sum(1 for r in rows if "error" not in r)
        # Returning "ok" when NOTHING succeeded is what let a wrong-organism
        # biosample produce a complete, empty Section 13 instead of an error
        # (2026-09-23).
        _status = "ok"
        _msg = ""
        if items and _n_ok == 0:
            _status = "error"
            _msg = ("all %d AlphaGenome item(s) failed for biosample %r / "
                    "organism %r -- no validation data was produced. First "
                    "error: %s" % (len(items), tissue_curie, organism,
                                   (errors[0] if errors else "?")))
        return json.dumps(
            {
                "status":         _status,
                **({"message": _msg} if _msg else {}),
                "cell_line":      cell_line,
                "tissue":         tissue_curie,
                "outputs":        requested_outputs,
                "outputs_dropped": _dropped_outputs,
                "access_modality": access_key,
                "window":         window,
                "alt_strategy":   alt_strategy,
                "n_items":        len(items),
                "n_ok":           _n_ok,
                "n_error":        len(errors),
                "total_elapsed_s": round(time.time() - t0, 2),
                "by_tf":          by_tf,
                "rows":           rows,
                "errors":         errors,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
