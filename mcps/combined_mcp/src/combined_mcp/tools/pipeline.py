"""combine_pvalues + pick_topn — deterministic Step 1->2 bridge."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from mcp.server.fastmcp import FastMCP


# Canonical INTEGRATED_ranked.csv schema -- matches the reference script 1.combined_rank.py
# (2026-06-04 update). Changes from 2026-05-27:
#   - REMOVED: "Zcom_signed" (signed Z was redundant with Combined Score)
#   - ADDED:   "Combined Score (Avg.Z)" = -log10(Pcom) * sign(log2FC)
#   - ADDED:   "Rank_Combined_Score" (1-based, after Combined Score desc sort)
#   - RENAMED: "Rank" -> "Rank_Pcom"
INTEGRATED_COLUMNS = [
    "Gene",
    "differential_MDscore",
    "log2FC",
    "ATAC_pvalue",
    "RNA_pvalue",
    "Zcom",
    "Pcom",
    "Combined Score (Avg.Z)",
    "Rank_Combined_Score",
    "Rank_Pcom",
]
# Backward-compat: old INTEGRATED.csv files (pre  2026-05-27 rename) only
# carried the first 6 cols. Downstream pick_topn only needs Gene+log2FC+Pcom,
# so we validate against this minimal set.
INTEGRATED_REQUIRED_COLUMNS = [
    "Gene", "differential_MDscore", "log2FC",
    "ATAC_pvalue", "RNA_pvalue", "Pcom",
]

# Columns that get min-max scaled by minmax_scale_integrated (reference script 1.minman.R)
SCALE_COLUMNS = ["differential_MDscore", "log2FC", "ATAC_pvalue", "RNA_pvalue"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
# HOCOMOCO v11 short-name -> HGNC gene symbol mapping.
# DAStk uses HOCOMOCO motif IDs like "ANDR_HUMAN.H11MO.0.A" -- the prefix
# before the first underscore is HOCOMOCO's TF "short name" which is NOT
# always the gene symbol. For example: ANDR=AR/NR3C4 (androgen receptor),
# AP2A=TFAP2A, GCR=NR3C1 (glucocorticoid receptor), COT1=NR2F1, TWST1=TWIST1.
# Without this mapping, those TFs are silently dropped during the join with
# DESeq2 (which uses real gene symbols).
HOCOMOCO_TO_GENE = {
    # 2026-09-11: short names of the DAStk refill beds (REFILL_20260910) that were
    # absent here, so their MD scores reached INTEGRATED.csv only as unmatched ids
    "ERR2": "ESRRB", "ERR3": "ESRRG", "HXA1": "HOXA1", "HXB13": "HOXB13", "HXB4": "HOXB4",
    "HXB7": "HOXB7", "HXC9": "HOXC9", "NDF2": "NEUROD2", "NF2L1": "NFE2L1", "NFAC2": "NFATC2",
    "NFAC3": "NFATC3", "NFAC4": "NFATC4", "NKX25": "NKX2-5", "NKX28": "NKX2-8", "NKX32": "NKX3-2",
    "STA5A": "STAT5A", "ZBT7A": "ZBTB7A",
    # 2026-10-12: short names of non-empty DAStk motif BEDs (grch38_motifs, mm10_motifs) that are not gene symbols
    # and were absent here, so the TF was dropped by the inner join with the expression table.
    # Symbol = the HOCOMOCO v11 model's UniProt entry (DAStk HOCOMOCOv11_to_uniprot.txt, MGI/HGNC
    # Swiss-Prot ids, UniProt) in the GENCODE v34 spelling (BMAL1 -> ARNTL).
    "ARI5B": "ARID5B", "ATF6A": "ATF6", "BMAL1": "ARNTL", "ERR1": "ESRRA", "EVI1": "MECOM", "HNF6": "ONECUT1",
    "PKNX1": "PKNOX1", "PRD14": "PRDM14", "RORG": "RORC", "Z324A": "ZNF324", "Z354A": "ZNF354A", "ZKSC1": "ZKSCAN1",
    # mouse-only models (mm10_motifs): human-style symbol, as every key here (the mouse join maps Nkx2-2 -> NKX2-2)
    "DMRTB": "DMRTB1", "MYBA": "MYBL1", "NGN2": "NEUROG2", "NKX22": "NKX2-2", "PRD16": "PRDM16", "TF2L1": "TFCP2L1",
    # Nuclear receptors / hormone receptors
    "ANDR":  "AR",        # androgen receptor (also referred to as NR3C4)
    "GCR":   "NR3C1",     # glucocorticoid receptor (= "GRE" motif)
    "PRGR":  "PGR",       # progesterone receptor (NR3C3)
    "COT1":  "NR2F1",     # COUP-TF I  (= NR2F1)
    "COT2":  "NR2F2",     # COUP-TF II
    "STF1":  "NR5A1",     # steroidogenic factor 1
    "THA":   "THRA",
    "THB":   "THRB",
    "THA11": "THAP11",
    # AP-2 family
    "AP2A":  "TFAP2A",    # = "AP-2alpha"
    "AP2B":  "TFAP2B",
    "AP2C":  "TFAP2C",
    # bHLH / TWIST / NEUROD
    "TWST1": "TWIST1",    # = TWIST1
    "BHA15": "BHLHA15",
    "BHE40": "BHLHE40",
    "HEN1":  "NHLH1",
    "NDF1":  "NEUROD1",
    "HTF4":  "TCF12",     # E2-2 = TCF4 (some references), but HTF4 -> TCF12 in HOCOMOCO
    "ITF2":  "TCF4",
    "TFE2":  "TCF3",      # E2A
    "LYL1":  "LYL1",
    # POU domain
    "PO5F1": "POU5F1",    # OCT4
    "PO2F1": "POU2F1",
    "PO2F2": "POU2F2",
    "PO3F1": "POU3F1",
    "PO3F2": "POU3F2",
    "PIT1":  "POU1F1",
    # BCL11 / HOX abbreviations
    "BC11A": "BCL11A",
    "HXA9":  "HOXA9",
    "HXA10": "HOXA10",
    "HXA13": "HOXA13",
    "HXB8":  "HOXB8",
    # NKX abbreviations (HOCOMOCO drops the hyphen)
    "NKX21": "NKX2-1",
    "NKX31": "NKX3-1",
    "NKX61": "NKX6-1",
    # p53 family
    "P53":   "TP53",
    "P63":   "TP63",
    "P73":   "TP73",
    # SMARCA / SREBF / Other
    "SMCA1": "SMARCA1",
    "SMCA5": "SMARCA5",
    "SRBP1": "SREBF1",
    "SRBP2": "SREBF2",
    "TYY1":  "YY1",
    "COE1":  "EBF1",
    "KAISO": "ZBTB33",
    "SUH":   "RBPJ",
    "PEBB":  "CBFB",
    "NF2L2": "NFE2L2",
    "NFAC1": "NFATC1",
    "TF65":  "RELA",
    "TF7L1": "TCF7L1",
    "TF7L2": "TCF7L2",
    "OZF":   "ZNF146",
    "STA5B": "STAT5B",
    # BRAC (Brachyury, recently renamed TBXT in HGNC)
    "BRAC":  "TBXT",
    # ZNF prefix abbreviation (HOCOMOCO uses ZN<num> for many)
    # We expand the common subset; remaining ZN* fall through to verbatim
    "ZN143": "ZNF143",
    "ZN263": "ZNF263",
    "ZN281": "ZNF281",
    "ZN384": "ZNF384",
    "ZN335": "ZNF335",
    "ZN670": "ZNF670",
    "ZN770": "ZNF770",
    # ZBT family
    "ZBT14": "ZBTB14",
    "ZBT17": "ZBTB17",
    "ZBT18": "ZBTB18",
    "ZBT48": "ZBTB48",
    "ZBTB6": "ZBTB6",
    # ZSC family
    "ZSC22": "ZSCAN22",
    "ZSC31": "ZSCAN31",
}


def _hocomoco_to_symbol(short: str) -> str:
    """Convert HOCOMOCO motif short name -> HGNC gene symbol.

    Returns the mapped symbol if known, else passes the short name through
    (covers cases where short == symbol, e.g. ETV1, GATA3).

    Also handles generic ZN<num> -> ZNF<num> when not in the explicit table.
    """
    if short in HOCOMOCO_TO_GENE:
        return HOCOMOCO_TO_GENE[short]
    # HOCOMOCO HOX short names: HXA13 -> HOXA13, HXB13 -> HOXB13 ...
    if len(short) > 3 and short[:2] == "HX" and short[2] in "ABCD" and short[3:].isdigit():
        return "HOX" + short[2:]
    # Heuristic: ZN<digits> -> ZNF<digits> (HOCOMOCO drops the 'F')
    if len(short) > 2 and short.startswith("ZN") and short[2].isdigit():
        return "ZNF" + short[2:]
    return short



def _read_dastk(atac_md_path: str) -> pd.DataFrame:
    """Parse DAStk ``*_differential_md_scores.txt`` (TSV, no header).

    HOCOMOCO motif id is split on the first underscore: ``REST_HUMAN.H11MO.0.A``
    -> TF gene = ``REST``. Multiple motif versions per gene are deduped, keeping
    the first occurrence — DAStk pre-sorts by ATAC p-value ascending, so this
    yields the strongest motif call per TF gene.
    """
    df = pd.read_csv(
        atac_md_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 7],
        names=["motif_id", "atac_pvalue", "diff_md_score"],
    )
    df["_short"] = df["motif_id"].str.split("_", n=1).str[0]
    df["Gene"] = df["_short"].apply(_hocomoco_to_symbol)
    df = df.drop_duplicates(subset="Gene", keep="first").reset_index(drop=True)
    return df[["Gene", "atac_pvalue", "diff_md_score"]]


def _is_human(organism) -> bool:
    return str(organism or "human").strip().lower() in (
        "human", "homo sapiens", "homo_sapiens", "hg38", "grch38", "hs", "9606")


def _ortholog_map(organism):
    """The orchestrator's mouse->human ortholog map, or None.

    combined_mcp is a standalone MCP package, so it reaches the orchestrator's
    `orthology` module by path rather than by dependency; that module imports
    stdlib only. None means "no table installed" and the caller falls back to
    case folding alone, which is still better than an exact-string join.
    """
    if _is_human(organism):
        return None
    try:
        import sys as _sys
        _src = os.environ.get("MRAG_ORCHESTRATOR_SRC",
                              "/BIO1/MRAG/Agent/orchestrator/src")
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from mrag_orchestrator.orthology import get_map as _get_map
        m = _get_map(str(organism).strip().lower())
        return m if (m is not None and len(m)) else None
    except Exception as e:          # never let the join die on a missing table
        print("[combine] ortholog map unavailable (%s: %s); falling back to "
              "case folding only" % (type(e).__name__, str(e)[:120]), flush=True)
        return None


def _join_key(series: pd.Series, organism: str, omap=None) -> pd.Series:
    """Key used to join the ATAC (motif) side to the RNA (expression) side.

    The ATAC side always carries HOCOMOCO TF names, which are UPPERCASE and
    HUMAN-derived for every species ("REST_MOUSE.H11MO.0.A" -> "REST", and
    HOCOMOCO's mouse models keep the human-style spelling). The RNA side
    carries the annotation's own symbols: uppercase for human, TitleCase for
    mouse ("Rest"). Joining those on the exact string returns zero rows, and a
    zero-row INTEGRATED.csv is the failure that produced an empty report
    instead of an error (2026-09-23).

    Two levels, because case folding alone is not enough:
      * upper-casing rescues Ar -> AR, Onecut2 -> ONECUT2, Rest -> REST;
      * the MGI ortholog map rescues the genuine renames that upper-casing
        cannot touch -- Trp53 -> TP53, Trp63 -> TP63, Nkx2-1 stays NKX2-1.
    A symbol with no ortholog keeps its uppercased self as the key (best
    effort for a JOIN; the LITERATURE layer treats "no ortholog" very
    differently -- see mrag_orchestrator.orthology).

    Human keeps the exact-string join so an existing run is bit-identical.
    """
    if _is_human(organism):
        return series
    up = series.astype(str).str.upper()
    if omap is None:
        return up
    return series.astype(str).map(lambda g: omap.to_human(g).join_key)


def _read_deseq(rna_deseq_path: str) -> pd.DataFrame:
    """Parse DESeq2 gene-level CSV.

    Expected columns: ID, GENE, baseMean, log2FoldChange, lfcSE, stat,
    pvalue, padj. Rows with NaN pvalue are dropped.
    """
    df = pd.read_csv(rna_deseq_path, usecols=["GENE", "log2FoldChange", "pvalue"])
    df = df.dropna(subset=["pvalue"])
    df = df.rename(
        columns={"GENE": "Gene", "log2FoldChange": "log2FC", "pvalue": "rna_pvalue"}
    )
    return df[["Gene", "log2FC", "rna_pvalue"]]


def _stouffer_combine(p_atac: pd.Series, p_rna: pd.Series):
    """Combine two p-value series with Stouffer's Z-method (equal weights).

    Returns ``(Z_com, P_com)`` as numpy float64 arrays aligned to the input
    index.  The reference script 1.combined_rank.py (2026-05-27) re-introduced Zcom alongside
    Zcom_signed (= Zcom * sign(log2FC)) and a 1-based Rank column.

    Note: the reference script uses ``stats.norm.ppf(1 - p)`` with p clipped to [1e-15, 1-1e-15];
    we use ``stats.norm.isf(p)`` (numerically equivalent at small p; its clip
    only matters when p == 0.0 exactly). Both produce identical Z within
    floating-point tolerance for non-degenerate inputs.
    """
    p_atac = np.clip(np.asarray(p_atac, dtype=float), 1e-15, 1 - 1e-15)
    p_rna  = np.clip(np.asarray(p_rna,  dtype=float), 1e-15, 1 - 1e-15)
    z_atac = stats.norm.isf(p_atac)
    z_rna  = stats.norm.isf(p_rna)
    z_com  = (z_atac + z_rna) / np.sqrt(2.0)
    p_com  = stats.norm.sf(z_com)
    return z_com, p_com


def _topn_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic ordering for Top-N selection.

    Primary key   : Pcom ascending (smaller p = better)
    Tiebreaker 1  : |log2FC| descending (RNA effect size)
    Tiebreaker 2  : |differential_MDscore| descending (ATAC effect size)
    Final tiebreak: Gene ascending (alphabetical) -- guarantees full determinism
    """
    out = df.copy()
    out["_absL2FC"] = out["log2FC"].abs()
    out["_absMD"] = out["differential_MDscore"].abs()
    out = out.sort_values(
        by=["Pcom", "_absL2FC", "_absMD", "Gene"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).drop(columns=["_absL2FC", "_absMD"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------
def register_pipeline_tools(mcp: FastMCP):

    @mcp.tool()
    def combine_pvalues(
        atac_md_path: str,
        rna_deseq_path: str,
        output_path: str,
        organism: str = "human",
        min_join_rows: int = 5,
        min_join_frac: float = 0.05,
    ) -> str:
        """Combine ATAC + RNA p-values per TF using Stouffer's Z-method.

        Args:
            atac_md_path:  Absolute path to DAStk
                ``*_differential_md_scores.txt``. TSV, no header;
                col1=motif_id, col2=ATAC pvalue, col8=differential MD-score.
            rna_deseq_path: Absolute path to DESeq2 gene-level CSV with
                columns ID,GENE,baseMean,log2FoldChange,lfcSE,stat,pvalue,padj.
            output_path: Absolute path where INTEGRATED.csv will be written.
            organism: "human" (default, exact symbol join -- unchanged) or
                "mouse" (case-folded join PLUS the MGI mouse->human ortholog
                map, so Ar/AR and Trp53/TP53 both match).
            min_join_rows: absolute floor on the joined row count (5). Below
                it the tool RAISES instead of writing a near-empty
                INTEGRATED.csv that every later section would silently
                inherit.
            min_join_frac: the same floor expressed as a fraction of the ATAC
                motif TFs (0.05). The effective floor is the larger of the
                two. Set both to 0 to disable (not recommended).

        Output CSV schema (reference script 1.combined_rank.py, 2026-05-27 schema):
            Gene, differential_MDscore, log2FC, ATAC_pvalue, RNA_pvalue,
            Zcom, Zcom_signed, Pcom, Rank

        Returns a JSON summary string with row counts and a top-5 preview
        (sorted by Pcom ascending with deterministic tiebreakers).
        """
        atac_md_path = os.fspath(Path(atac_md_path).expanduser())
        rna_deseq_path = os.fspath(Path(rna_deseq_path).expanduser())
        out = Path(output_path).expanduser()

        atac = _read_dastk(atac_md_path)
        rna = _read_deseq(rna_deseq_path)
        _omap = _ortholog_map(organism)
        # The ATAC side is already the human-style HOCOMOCO spelling, so it is
        # only case-folded; the RNA side is the one that needs translating.
        atac["_join"] = _join_key(atac["Gene"], organism)
        rna["_join"] = _join_key(rna["Gene"], organism, omap=_omap)

        # Several mouse genes can share ONE human ortholog (Mycn and Mycs both
        # -> MYCN), and case folding collapses more. pd.merge then duplicates
        # the ATAC row, so INTEGRATED.csv carried the same TF twice with
        # contradictory statistics and a "top 4" panel held 3 distinct TFs
        # (2026-09-23). Collapse the expression side FIRST, deterministically:
        # the gene that carries the join key's own name wins, then the smallest
        # RNA p-value, then alphabetical order. Human is untouched -- its join
        # key is the exact symbol, so this path never runs and the output stays
        # bit-identical.
        _collapsed: list = []
        if not _is_human(organism):
            _dup = set(rna.loc[rna["_join"].duplicated(keep=False), "_join"]
                       .astype(str))
            if _dup:
                _rn = rna.assign(
                    _notself=(rna["Gene"].astype(str).str.upper()
                              != rna["_join"].astype(str)).astype(int),
                    _p=pd.to_numeric(rna["rna_pvalue"], errors="coerce")
                        .fillna(1.0),
                    _g=rna["Gene"].astype(str))
                _rn = _rn.sort_values(["_join", "_notself", "_p", "_g"],
                                      kind="mergesort")
                for _k, _grp in _rn[_rn["_join"].astype(str).isin(_dup)
                                    ].groupby("_join", sort=True):
                    _collapsed.append({
                        "join_key": str(_k),
                        "kept": str(_grp["Gene"].iloc[0]),
                        "dropped": [str(x) for x in _grp["Gene"].iloc[1:]]})
                rna = (_rn.drop_duplicates(subset="_join", keep="first")
                          .drop(columns=["_notself", "_p", "_g"])
                          .reset_index(drop=True))
                print("[combine] %d join key(s) had several expression rows; "
                      "kept one each (e.g. %s)"
                      % (len(_collapsed),
                         ", ".join("%s<-%s" % (c["kept"], "/".join(c["dropped"]))
                                   for c in _collapsed[:5])), flush=True)

        # Keep the ATAC (HOCOMOCO) spelling of the TF name, as before, but carry
        # the RNA-side spelling alongside it: every downstream consumer that
        # re-joins INTEGRATED.csv to DESeq2.gene.csv on the symbol needs the
        # RUN's namespace, not the motif namespace (Section 7's heatmaps matched
        # nothing on mouse and rendered empty -- 2026-09-23).
        merged = pd.merge(atac, rna.rename(columns={"Gene": "RNA_Gene"}),
                          on="_join", how="inner").drop(columns=["_join"])

        # A join that returns 0 rows -- or a handful -- is not a result. Both
        # are the same failure (a symbol-namespace mismatch) seen at different
        # noise levels, and both used to sail through: empty returned status
        # "warning", and 3-rows-out-of-400 returned status "ok". Everything
        # downstream is built from INTEGRATED.csv, so both produced a complete,
        # plausible-looking, wrong report.
        _floor = max(int(min_join_rows or 0),
                     int(math.ceil(float(min_join_frac or 0.0) * len(atac))))
        if merged.empty or len(merged) < _floor:
            # NOT a warning. Every downstream section is built from
            # INTEGRATED.csv, so returning "warning" here produced a complete,
            # empty, plausible-looking report. The commonest cause is a symbol
            # namespace mismatch (uppercase HOCOMOCO vs TitleCase mouse
            # symbols, or Ensembl ids that were never mapped to symbols at all).
            _rna_head = [str(x) for x in rna["Gene"].head(8).tolist()]
            _atac_head = [str(x) for x in atac["Gene"].head(8).tolist()]
            _joined_now = set(merged["Gene"].astype(str)) if len(merged) else set()
            _atac_miss = [g for g in atac["Gene"].astype(str)
                          if g not in _joined_now][:8]
            _rna_keys = set(rna["_join"].astype(str))
            _rna_miss = [g for g, k in zip(rna["Gene"].astype(str),
                                           rna["_join"].astype(str))
                         if k not in set(atac["_join"].astype(str))][:8]
            raise ValueError(
                "ATAC/RNA join produced %d row(s), below the floor of %d "
                "(min_join_rows=%s, min_join_frac=%s of %d ATAC motif TFs), "
                "under organism=%r%s.\n"
                "  ATAC MD-score table : %s -- %d TF rows, e.g. %s\n"
                "  DESeq2 table        : %s -- %d gene rows, e.g. %s\n"
                "  ATAC TFs that matched nothing : %s\n"
                "  RNA genes that matched nothing: %s\n"
                "Nothing downstream can be computed from this: every report "
                "section is derived from INTEGRATED.csv, so continuing would "
                "produce a complete but empty report instead of an error. "
                "Commonest causes: the RNA table carries bare ENSG/ENSMUSG ids "
                "instead of SYMBOLS; the two sides are different species; or a "
                "non-human run was left at organism=\"human\", which uses the "
                "exact-string join."
                % (len(merged), _floor, min_join_rows, min_join_frac,
                   len(atac), organism,
                   ("" if _omap is None else
                    " (ortholog map: %s, %d symbols)"
                    % (_omap.source, len(_omap))),
                   atac_md_path, len(atac), _atac_head,
                   rna_deseq_path, len(rna), _rna_head,
                   _atac_miss, _rna_miss))

        # Clip p-values to [1e-15, 1-1e-15] BEFORE storing — 
        # 1.combined_rank.py applies this floor/ceiling both for Stouffer
        # combination AND for the output columns. We mirror exactly.
        atac_p_clipped = np.clip(np.asarray(merged["atac_pvalue"], dtype=float),
                                 1e-15, 1 - 1e-15)
        rna_p_clipped  = np.clip(np.asarray(merged["rna_pvalue"],  dtype=float),
                                 1e-15, 1 - 1e-15)
        z_com, p_com = _stouffer_combine(atac_p_clipped, rna_p_clipped)
        # Reference script 1.combined_rank.py (2026-06-04): Combined Score = -log10(Pcom) * sign(log2FC).
        # This signed score is the primary ranking key (replaces Zcom_signed).
        sign = np.sign(np.asarray(merged["log2FC"], dtype=float))
        combined_score = -np.log10(p_com) * sign
        _frame = {
                "Gene":                merged["Gene"],
                "differential_MDscore": merged["diff_md_score"],
                "log2FC":              merged["log2FC"],
                "ATAC_pvalue":         atac_p_clipped,
                "RNA_pvalue":          rna_p_clipped,
                "Zcom":                z_com,
                "Pcom":                p_com,
                "Combined Score (Avg.Z)": combined_score,
        }
        # HUMAN KEEPS ITS EXACT SCHEMA. The extra column exists only where the
        # two namespaces actually differ, so an existing human INTEGRATED.csv
        # is byte-identical to before.
        _out_cols = list(INTEGRATED_COLUMNS)
        if not _is_human(organism):
            _frame["RNA_Gene"] = merged["RNA_Gene"]
            _out_cols = _out_cols + ["RNA_Gene"]
        result = pd.DataFrame(_frame)
        # Rank_Combined_Score: 1-based after Combined Score DESC sort.
        result = result.sort_values("Combined Score (Avg.Z)",
                                    ascending=False).reset_index(drop=True)
        result["Rank_Combined_Score"] = np.arange(1, len(result) + 1)
        # Rank_Pcom: 1-based after Pcom ASC sort (smaller p = better).
        result = result.sort_values("Pcom", ascending=True).reset_index(drop=True)
        result["Rank_Pcom"] = np.arange(1, len(result) + 1)
        result = result[_out_cols]

        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)

        preview = _topn_sort_key(result).head(5).to_dict(orient="records")
        # Which ATAC TFs found no expression row. Case folding rescues symbols
        # that differ only in case (REST/Rest) but NOT genuine orthologue
        # renames (human TP53 vs mouse Trp53), so say how many were lost rather
        # than let the panel quietly shrink.
        _joined = set(merged["Gene"].astype(str))
        _unjoined = [g for g in atac["Gene"].astype(str) if g not in _joined]
        if _unjoined and len(_unjoined) > 0.2 * max(len(atac), 1):
            print("[combine] WARNING: %d/%d ATAC motif TFs matched no gene in the "
                  "expression table (organism=%s). First few: %s"
                  % (len(_unjoined), len(atac), organism, _unjoined[:10]), flush=True)
        return json.dumps(
            {
                "status": "ok",
                "output": str(out),
                "atac_tfs": int(len(atac)),
                "rna_genes": int(len(rna)),
                "integrated_rows": int(len(result)),
                "unjoined_atac_tfs": int(len(_unjoined)),
                "unjoined_examples": _unjoined[:10],
                "organism": organism,
                "join_floor": int(_floor),
                "collapsed_rna_duplicates": len(_collapsed),
                "collapsed_examples": _collapsed[:10],
                "rna_gene_column": bool(not _is_human(organism)),
                "ortholog_map": ("" if _omap is None else _omap.source),
                "ortholog_symbols": (0 if _omap is None else len(_omap)),
                "top5_by_pcom": preview,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def pick_topn(
        integrated_csv: str,
        output_dir: str,
        n_list: Optional[List[int]] = None,
    ) -> str:
        """Pick Top-N TFs from INTEGRATED.csv for the Step 2 fork.

        Args:
            integrated_csv: Path to INTEGRATED.csv from combine_pvalues.
            output_dir:     Directory where per-N outputs are written.
            n_list:         List of cutoffs (default ``[4, 6, 8]``).

        For each N this writes two files inside ``output_dir``:
            ``top{N}.json`` -- ``{"n": N, "tfs": [...], "details": [...]}``
            ``top{N}.tsv``  -- tab-separated table of the N rows

        Also writes ``topN_union.json`` -- union of all Top-N sets (used by
        AlphaGenome validation + paper-qa retrieval).

        Returns a JSON summary string with per-N TF lists and the union.
        """
        if n_list is None:
            n_list = [4, 6, 8]

        integrated_csv = Path(integrated_csv).expanduser()
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(integrated_csv)
        if df.empty:
            # Same class of silent failure as the join: an empty INTEGRATED.csv
            # used to yield top4.json/top6.json with empty "tfs" lists and
            # status "ok", and every later section then rendered itself empty.
            raise ValueError(
                "%s has no rows, so no Top-N TF panel can be picked. An empty "
                "INTEGRATED.csv means the ATAC/RNA join found no shared TF "
                "genes; fix that rather than continuing with an empty panel."
                % integrated_csv)
        missing = [c for c in INTEGRATED_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"INTEGRATED.csv missing columns: {missing}",
                    "found": list(df.columns),
                },
                ensure_ascii=False,
                indent=2,
            )

        # Ranking column precedence (2026-06-04 update):
        #   1. Rank_Combined_Score   <- canonical (signed Combined Score desc)
        #   2. Rank_Pcom             <- legacy single-tail (Pcom asc)
        #   3. Rank                  <- pre-2026-06-04 alias for Rank_Pcom
        #   4. _topn_sort_key()      <- deterministic fallback
        # Combined Score is now the primary key because it preserves direction
        # (upregulated vs downregulated) and is the basis for the reference analysis's new
        # rank/volcano/heatmap plots.
        if "Rank_Combined_Score" in df.columns:
            sorted_df = df.sort_values("Rank_Combined_Score",
                                       ascending=True).reset_index(drop=True)
        elif "Rank_Pcom" in df.columns:
            sorted_df = df.sort_values("Rank_Pcom",
                                       ascending=True).reset_index(drop=True)
        elif "Rank" in df.columns:
            sorted_df = df.sort_values("Rank",
                                       ascending=True).reset_index(drop=True)
        else:
            sorted_df = _topn_sort_key(df)
        # Panel = TFs whose motif is MORE active in the treatment arm
        # (differential_MDscore >= 0). Negative-MDscore TFs are more active in
        # the control arm -> not drivers of the treatment program; drop them
        # before the top-N pick (2026-07 directive). Safety: if the filter would
        # empty the frame, keep the unfiltered order.
        if "differential_MDscore" in sorted_df.columns:
            _pos = sorted_df[sorted_df["differential_MDscore"].fillna(0) >= 0]
            if len(_pos) > 0:
                sorted_df = _pos.reset_index(drop=True)
        result: dict = {
            "status": "ok",
            "input": str(integrated_csv),
            "n_list": sorted(set(int(x) for x in n_list)),
            "by_n": {},
        }
        union: set = set()

        # A panel shorter than requested is a real, quiet degradation: every
        # downstream section is sized by the panel, so "top 10" silently
        # becoming "top 3" changes the whole report with nothing to show for it.
        # DISTINCT TFs, not rows. A duplicated Gene made a top-4 panel hold
        # 3 distinct TFs while len(tfs) was still 4, so this warning never
        # fired for exactly the case it was written for (2026-09-23).
        _n_rows = len(sorted_df)
        _n_distinct = int(sorted_df["Gene"].astype(str).nunique())
        _usable = min(_n_rows, _n_distinct)
        _short = [n for n in result["n_list"] if n > _usable]
        if _short:
            result["short_panels"] = {int(n): int(_usable) for n in _short}
            result["distinct_tfs"] = _n_distinct
            result["warning"] = (
                "INTEGRATED.csv has only %d usable row(s) (%d DISTINCT TF%s) "
                "after the positive-MD-score filter, so top-N for N in %s is "
                "short."
                % (_n_rows, _n_distinct, "" if _n_distinct == 1 else "s",
                   _short))
            print("[pick_topn] WARNING: %s" % result["warning"], flush=True)
        if _n_distinct < _n_rows:
            _dups = sorted(set(sorted_df["Gene"].astype(str)[
                sorted_df["Gene"].astype(str).duplicated(keep=False)]))
            result["duplicate_tfs"] = _dups[:20]
            print("[pick_topn] WARNING: %d of %d rows are repeats of the same "
                  "TF (%s); a top-N panel will hold fewer than N distinct TFs."
                  % (_n_rows - _n_distinct, _n_rows, _dups[:10]), flush=True)
        for n in result["n_list"]:
            head = sorted_df.head(n).copy()
            head_records = head.to_dict(orient="records")
            tfs = head["Gene"].tolist()
            payload = {"n": n, "tfs": tfs, "details": head_records}
            json_path = out_dir / f"top{n}.json"
            tsv_path = out_dir / f"top{n}.tsv"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            )
            head.to_csv(tsv_path, sep="\t", index=False)
            result["by_n"][str(n)] = {
                "tfs": tfs,
                "details": head_records,
                "json": str(json_path),
                "tsv": str(tsv_path),
            }
            union.update(tfs)

        union_sorted = sorted(union)
        union_path = out_dir / "topN_union.json"
        union_path.write_text(
            json.dumps(
                {"n_list": result["n_list"], "tfs": union_sorted},
                ensure_ascii=False,
                indent=2,
            )
        )
        result["union"] = {
            "size": len(union_sorted),
            "tfs": union_sorted,
            "json": str(union_path),
        }
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    @mcp.tool()
    def combine_and_pick(
        atac_md_path: str,
        rna_deseq_path: str,
        output_dir: str,
        n_list: Optional[List[int]] = None,
        organism: str = "human",
    ) -> str:
        """Composite: combine_pvalues + pick_topn in one call.

        Writes ``{output_dir}/INTEGRATED.csv`` and ``{output_dir}/top{N}.{json,tsv}``.
        """
        if n_list is None:
            n_list = [4, 6, 8]
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        integrated_path = out_dir / "INTEGRATED.csv"

        combine_result = json.loads(
            combine_pvalues(atac_md_path, rna_deseq_path, str(integrated_path),
                            organism=organism)
        )
        if combine_result.get("status") != "ok":
            return json.dumps(combine_result, ensure_ascii=False, indent=2)

        pick_result = json.loads(pick_topn(str(integrated_path), str(out_dir), n_list))
        # pick_topn's own failure used to be wrapped in an outer status "ok".
        # CombinedNode reads only the OUTER status, so a top-N pick that failed
        # on a schema mismatch was reported as a successful run with empty
        # panels (2026-09-23).
        if pick_result.get("status") != "ok":
            return json.dumps({"status": pick_result.get("status", "error"),
                               "message": "top-N pick failed after a successful "
                                          "combine; INTEGRATED.csv was written "
                                          "but no TF panel could be selected",
                               "combine": combine_result,
                               "pick_topn": pick_result},
                              ensure_ascii=False, indent=2, default=str)
        return json.dumps(
            {
                "status": "ok",
                "combine": combine_result,
                "pick_topn": pick_result,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def integrated_summary(integrated_csv: str, top_k: int = 10) -> str:
        """Quick summary of an existing INTEGRATED.csv.

        Returns row count, column list, Pcom min/median, and the top-K rows
        (sorted by Pcom ascending with deterministic tiebreakers).
        """
        df = pd.read_csv(Path(integrated_csv).expanduser())
        # Ranking column precedence (2026-06-04 update):
        #   1. Rank_Combined_Score   <- canonical (signed Combined Score desc)
        #   2. Rank_Pcom             <- legacy single-tail (Pcom asc)
        #   3. Rank                  <- pre-2026-06-04 alias for Rank_Pcom
        #   4. _topn_sort_key()      <- deterministic fallback
        # Combined Score is now the primary key because it preserves direction
        # (upregulated vs downregulated) and is the basis for the reference analysis's new
        # rank/volcano/heatmap plots.
        if "Rank_Combined_Score" in df.columns:
            sorted_df = df.sort_values("Rank_Combined_Score",
                                       ascending=True).reset_index(drop=True)
        elif "Rank_Pcom" in df.columns:
            sorted_df = df.sort_values("Rank_Pcom",
                                       ascending=True).reset_index(drop=True)
        elif "Rank" in df.columns:
            sorted_df = df.sort_values("Rank",
                                       ascending=True).reset_index(drop=True)
        else:
            sorted_df = _topn_sort_key(df)
        return json.dumps(
            {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pcom_min": float(df["Pcom"].min()),
                "pcom_median": float(df["Pcom"].median()),
                f"top{top_k}": sorted_df.head(top_k).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @mcp.tool()
    def minmax_scale_integrated(
        integrated_csv: str,
        output_csv: str,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Min-max scale the per-TF score columns of INTEGRATED.csv.

        Mirrors the reference script ``1.minman.R`` exactly:
            df_scaled = (df - min) / (max - min) for each named column.
            Output  = original columns + ``{column}_scaled`` columns.

        Args:
            integrated_csv: Path to INTEGRATED.csv (output of combine_pvalues).
            output_csv:     Where to write the scaled CSV.
            columns:        Columns to scale (default = the reference script's 4:
                            ``["differential_MDscore","log2FC",
                            "ATAC_pvalue","RNA_pvalue"]``).
        """
        if columns is None:
            columns = list(SCALE_COLUMNS)
        path = Path(integrated_csv).expanduser()
        if not path.is_file():
            return json.dumps({"status": "error", "message": f"not found: {path}"})
        out  = Path(output_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(path)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return json.dumps({
                "status":  "error",
                "message": f"columns missing in INTEGRATED.csv: {missing}",
                "found":   list(df.columns),
            }, ensure_ascii=False, indent=2)

        scaled = df.copy()
        for c in columns:
            x = pd.to_numeric(df[c], errors="coerce")
            mn, mx = x.min(), x.max()
            scaled[f"{c}_scaled"] = (x - mn) / (mx - mn) if mx > mn else x * 0.0

        scaled.to_csv(out, index=False)
        return json.dumps(
            {
                "status":         "ok",
                "input":          str(path),
                "output":         str(out),
                "n_rows":         int(len(df)),
                "scaled_columns": [f"{c}_scaled" for c in columns],
                "head":           scaled.head(3).to_dict(orient="records"),
            },
            ensure_ascii=False, indent=2, default=str,
        )

