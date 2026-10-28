"""build_tfs_pval3 -- HOMER filter + DiffBind + DESeq2 integration.

Mirrors ``target_TFS.py`` STEP4+STEP5:
  STEP4: per-peak DiffBind region overlap + TYPE 1-4 labelling.
  STEP5: per-gene RNA log2FC / pval join.

the reference intermediate ``Diffbind_gene.out`` (gene-name annotation of DiffBind
regions via HOMER annot.txt) is bypassed -- we match peak <-> DiffBind by
Chr + coordinate overlap only, since the HOMER filter output already
carries Gene Name per peak. In practice peaks that overlap on the same
chromosome are on the same locus.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re
from scipy import stats

from mcp.server.fastmcp import FastMCP


# ----  STEP4 TYPE definitions (peak vs DiffBind region) ---------------
def _overlap_type(peak_start: int, peak_end: int,
                  region_start: int, region_end: int) -> Optional[str]:
    """Return  overlap TYPE (1-4) or None when there is no overlap.

    Mirrors target_TFS.py STEP4_Diffbind_pval logic exactly (elif chain):
      TYPE1: peak overlaps region's LEFT edge  (peak.start <= region.start <= peak.end AND peak.end <= region.end)
      TYPE2: peak fully INSIDE region          (region.start <= peak.start AND peak.end <= region.end)
      TYPE3: peak overlaps region's RIGHT edge (peak.start <= region.end <= peak.end AND region.start <= peak.start)
      TYPE4: region fully INSIDE peak          (peak.start <= region.end <= peak.end AND peak.start <= region.start <= peak.end)
    """
    if (peak_start <= region_start <= peak_end) and (region_end >= peak_end):
        return "TYPE1"
    if (region_start <= peak_start <= region_end) and (region_start <= peak_end <= region_end):
        return "TYPE2"
    if (peak_start <= region_end <= peak_end) and (region_start <= peak_start):
        return "TYPE3"
    if (peak_start <= region_end <= peak_end) and (peak_start <= region_start <= peak_end):
        return "TYPE4"
    return None


def _index_diffbind_by_chr(diffbind_csv: Path) -> Dict[str, List[Tuple]]:
    """Read DiffBind_out.csv and group regions by chromosome.

    Returns: {chr: [(start, end, Fold, p.value, FDR), ...]}
    Sorted by region start ascending within each chr (binary search ready
    but we do linear scan -- O(n) per chr is fine for ~10-50 regions per
    typical chr-scoped query).
    """
    df = pd.read_csv(diffbind_csv)
    needed = ["seqnames", "start", "end", "Fold", "p.value", "FDR"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DiffBind_out.csv missing columns: {missing}")
    by_chr: Dict[str, List[Tuple]] = {}
    for chr_, sub in df[needed].groupby("seqnames"):
        sub = sub.sort_values("start")
        by_chr[str(chr_)] = list(
            zip(
                sub["start"].astype(int).tolist(),
                sub["end"].astype(int).tolist(),
                sub["Fold"].astype(float).tolist(),
                sub["p.value"].astype(float).tolist(),
                sub["FDR"].astype(float).tolist(),
            )
        )
    return by_chr


def _load_deseq_collapsed(
    deseq_csv: Path,
    gene_col: str = "GENE",
    fc_col: str = "log2FoldChange",
    pval_col: str = "pvalue",
    padj_col: str = "padj",
    collapse: str = "best_padj",
) -> Dict[str, Tuple[float, float, float]]:
    """Read DESeq2 CSV and collapse to one record per gene symbol.

    collapse strategies:
      best_padj : keep row with smallest padj per gene (NaN treated as +inf)
      best_pval : keep row with smallest pvalue per gene
      first     : keep first-seen row per gene (reference-analysis-like, raw order)
    """
    df = pd.read_csv(deseq_csv)
    if gene_col not in df.columns:
        raise ValueError(f"deseq csv missing '{gene_col}'; found {list(df.columns)}")
    # Drop NaN gene labels
    df = df[df[gene_col].notna() & (df[gene_col].astype(str).str.len() > 0)].copy()
    df[gene_col] = df[gene_col].astype(str)
    # Drop rows with no valid RNA p-value (mirrors the reference script 3.Diffbind_pval.py
    # DESeq2_filtered: a gene/isoform whose DESeq2 pvalue is NA carries no
    # usable RNA signal and must not enter the RNA join). Applied BEFORE the
    # per-gene collapse so a NA-pvalue isoform never shadows a valid one and
    # genes with only NA-pvalue isoforms are removed entirely.
    if pval_col in df.columns:
        df = df[pd.to_numeric(df[pval_col], errors="coerce").notna()].copy()

    if collapse == "best_padj" and padj_col in df.columns:
        df["_rank"] = pd.to_numeric(df[padj_col], errors="coerce").fillna(np.inf)
        df = df.sort_values([gene_col, "_rank"], kind="mergesort")
        df = df.drop_duplicates(subset=gene_col, keep="first").drop(columns="_rank")
    elif collapse == "best_pval":
        df["_rank"] = pd.to_numeric(df[pval_col], errors="coerce").fillna(np.inf)
        df = df.sort_values([gene_col, "_rank"], kind="mergesort")
        df = df.drop_duplicates(subset=gene_col, keep="first").drop(columns="_rank")
    else:  # "first" or fallback
        df = df.drop_duplicates(subset=gene_col, keep="first")

    out: Dict[str, Tuple[float, float, float]] = {}
    for g, fc, pv, pa in zip(
        df[gene_col],
        pd.to_numeric(df[fc_col], errors="coerce"),
        pd.to_numeric(df[pval_col], errors="coerce"),
        pd.to_numeric(df.get(padj_col, pd.Series([np.nan] * len(df))), errors="coerce"),
    ):
        out[str(g)] = (
            float(fc) if pd.notna(fc) else float("nan"),
            float(pv) if pd.notna(pv) else float("nan"),
            float(pa) if pd.notna(pa) else float("nan"),
        )
    return out


def register_integrate_tools(mcp: FastMCP):

    @mcp.tool()
    def build_tfs_pval3(
        homer_filter_tsv: str,
        diffbind_csv: str,
        deseq_csv: str,
        output_tsv: str,
        rna_collapse: str = "best_padj",
        keep_unmatched_diffbind: bool = False,
        keep_unmatched_rna: bool = False,
    ) -> str:
        """Integrate HOMER filter + DiffBind + DESeq2 into one evidence table.

        Mirrors target_TFS.py STEP4 (DiffBind overlap + TYPE 1-4) and
        STEP5 (RNA join) in one composite call. The single output mirrors
        the reference analysis's conceptual ``{sample}.TFs.Pval3.txt``.

        Args:
            homer_filter_tsv:   Output of mcp_homer.filter_target_tfs
                                (promoter peaks * selected TF columns).
            diffbind_csv:       Output of mcp_diffbind.run_diffbind
                                (DiffBind_out.csv).
            deseq_csv:          DESeq2 gene-level CSV with columns
                                ID, GENE, baseMean, log2FoldChange, lfcSE,
                                stat, pvalue, padj.
            output_tsv:         Where to write the integrated TSV.
            rna_collapse:       How to collapse multi-isoform RNA records
                                per gene: ``best_padj`` (default),
                               ``best_pval``, or ``first``.
            keep_unmatched_diffbind:
                                If False (default), drop peaks that have
                                no overlapping DiffBind region.
            keep_unmatched_rna: If False (default), drop rows whose Gene
                                Name has no DESeq2 record.

        Output schema:
            (all homer_filter_tsv columns)
            + DiffBind columns: Fold, DiffBind_Pval, DiffBind_FDR,
                                region_start, region_end, TYPE
            + RNA columns:      log2FC, RNA_pval, RNA_padj
        """
        homer_path = Path(homer_filter_tsv).expanduser()
        diffbind_path = Path(diffbind_csv).expanduser()
        deseq_path = Path(deseq_csv).expanduser()
        out_path = Path(output_tsv).expanduser()

        for p, label in ((homer_path, "homer"), (diffbind_path, "diffbind"), (deseq_path, "deseq")):
            if not p.is_file():
                return json.dumps({"status": "error",
                                   "message": f"{label} input not found: {p}"})

        # 1) Load HOMER filter TSV
        homer = pd.read_csv(homer_path, sep="\t", dtype=str)
        for col in ("PeakID", "Chr", "Start", "End", "Gene Name"):
            if col not in homer.columns:
                return json.dumps({
                    "status": "error",
                    "message": f"homer_filter_tsv missing '{col}'",
                    "found":   list(homer.columns),
                })
        n_peaks = len(homer)

        # 2) Index DiffBind by Chr
        try:
            db_by_chr = _index_diffbind_by_chr(diffbind_path)
        except ValueError as e:
            return json.dumps({"status": "error", "message": str(e)})

        # 3) Load DESeq collapsed
        try:
            rna = _load_deseq_collapsed(deseq_path, collapse=rna_collapse)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"deseq load failed: {e}"})

        # 4) STEP4: per-peak DiffBind overlap matching
        diffbind_records: List[Dict[str, Any]] = []
        peak_idx_with_match: List[int] = []
        for i, row in homer.iterrows():
            chr_ = row["Chr"]
            try:
                pstart = int(row["Start"])
                pend   = int(row["End"])
            except Exception:
                continue
            cands = db_by_chr.get(chr_, [])
            best = None
            # STEP4 takes the FIRST matching overlap via elif chain.
            # We mirror: linear scan, keep the first overlapping region.
            # If multiple regions overlap, we keep the one with smallest p.value
            # (more robust than positional first).
            for (rs, re_, fold, pv, fdr) in cands:
                # quick reject for non-overlap (saves time on long chrs)
                if re_ < pstart or rs > pend:
                    continue
                tp = _overlap_type(pstart, pend, rs, re_)
                if tp is None:
                    continue
                if best is None or pv < best["DiffBind_Pval"]:
                    best = {
                        "Fold":          fold,
                        "DiffBind_Pval": pv,
                        "DiffBind_FDR":  fdr,
                        "region_start":  rs,
                        "region_end":    re_,
                        "TYPE":          tp,
                    }
            if best is not None:
                diffbind_records.append(best)
                peak_idx_with_match.append(i)
            elif keep_unmatched_diffbind:
                diffbind_records.append({
                    "Fold": np.nan, "DiffBind_Pval": np.nan, "DiffBind_FDR": np.nan,
                    "region_start": np.nan, "region_end": np.nan, "TYPE": None,
                })
                peak_idx_with_match.append(i)

        merged = homer.loc[peak_idx_with_match].reset_index(drop=True)
        if diffbind_records:
            db_df = pd.DataFrame(diffbind_records)
            for c in db_df.columns:
                merged[c] = db_df[c].values
        else:
            for c in ["Fold", "DiffBind_Pval", "DiffBind_FDR", "region_start", "region_end", "TYPE"]:
                merged[c] = np.nan

        n_after_diffbind = len(merged)

        # 5) STEP5: per-gene RNA join (collapsed)
        log2FC, rna_pval, rna_padj = [], [], []
        for gene in merged["Gene Name"]:
            if gene in rna:
                fc, pv, pa = rna[gene]
                log2FC.append(fc); rna_pval.append(pv); rna_padj.append(pa)
            else:
                log2FC.append(np.nan); rna_pval.append(np.nan); rna_padj.append(np.nan)
        merged["log2FC"]   = log2FC
        merged["RNA_pval"] = rna_pval
        merged["RNA_padj"] = rna_padj

        if not keep_unmatched_rna:
            merged = merged[merged["log2FC"].notna()].reset_index(drop=True)
        n_after_rna = len(merged)

        # 6) Write
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, sep="\t", index=False)

        # 7) Summary
        unique_genes = merged["Gene Name"].nunique() if len(merged) else 0
        # TF columns = anything that isn't standard
        homer_base = {"PeakID","Chr","Start","End","Annotation","Distance to TSS","Gene Name"}
        non_tf = homer_base | {"Fold","DiffBind_Pval","DiffBind_FDR","region_start","region_end","TYPE",
                               "log2FC","RNA_pval","RNA_padj"}
        tf_cols = [c for c in merged.columns if c not in non_tf]

        # Per-TF non-empty counts (how many peaks each TF binds in final table)
        per_tf_n = {}
        for c in tf_cols:
            s = merged[c].astype(str).fillna("").replace("nan", "")
            per_tf_n[c] = int((s.str.len() > 0).sum())

        return json.dumps(
            {
                "status":              "ok",
                "input_homer":         str(homer_path),
                "input_diffbind":      str(diffbind_path),
                "input_deseq":         str(deseq_path),
                "output":              str(out_path),
                "n_peaks_input":       int(n_peaks),
                "n_after_diffbind":    int(n_after_diffbind),
                "n_after_rna":         int(n_after_rna),
                "n_unique_genes":      int(unique_genes),
                "tf_columns":          tf_cols,
                "peaks_per_tf":        per_tf_n,
                "rna_collapse":        rna_collapse,
                "kept_unmatched_diffbind": keep_unmatched_diffbind,
                "kept_unmatched_rna":      keep_unmatched_rna,
                "preview":             merged.head(3).to_dict(orient="records"),
            },
            ensure_ascii=False, indent=2, default=str,
        )

    @mcp.tool()
    def tfs_pval3_summary(tfs_pval3_tsv: str, top_k: int = 10) -> str:
        """Quick summary of an existing TFs.Pval3.txt produced by build_tfs_pval3.

        Returns row count, column list, FDR/pval distributions, and top-K
        rows by combined evidence (smaller DiffBind_FDR * RNA_padj).
        """
        path = Path(tfs_pval3_tsv).expanduser()
        if not path.is_file():
            return json.dumps({"status": "error", "message": f"not found: {path}"})
        df = pd.read_csv(path, sep="\t")
        result: Dict[str, Any] = {
            "status":  "ok",
            "input":   str(path),
            "n_rows":  int(len(df)),
            "columns": list(df.columns),
        }
        if "Gene Name" in df.columns:
            result["n_unique_genes"] = int(df["Gene Name"].nunique())
        if "TYPE" in df.columns:
            result["type_counts"] = df["TYPE"].value_counts().to_dict()
        for c in ("DiffBind_FDR", "RNA_padj"):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                result[c] = {
                    "min":    float(v.min()),
                    "median": float(v.median()),
                    "n_sig_0.05": int((v < 0.05).sum()),
                    "n_sig_0.01": int((v < 0.01).sum()),
                }
        # Top-K by combined evidence
        if {"DiffBind_FDR", "RNA_padj"} <= set(df.columns):
            df2 = df.copy()
            df2["_score"] = (-np.log10(pd.to_numeric(df2["DiffBind_FDR"], errors="coerce").clip(lower=1e-300))
                             + -np.log10(pd.to_numeric(df2["RNA_padj"], errors="coerce").clip(lower=1e-300)))
            top = df2.sort_values("_score", ascending=False).head(top_k).drop(columns="_score")
            result[f"top{top_k}_by_combined_logFDR"] = top.to_dict(orient="records")
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)


    @mcp.tool()
    def rank_target_genes(
        tfs_pval3_tsv: str,
        output_tsv: str,
        top_k_per_tf: int = 8,
        ranking: str = "stouffer",
        require_same_direction: bool = True,
        dedup_per_tf: bool = True,
        fold_col: str = "Fold",
        log2fc_col: str = "log2FC",
        fdr_col: str = "DiffBind_FDR",
        padj_col: str = "RNA_padj",
    ) -> str:
        """Per-TF top-K target gene ranking via Stouffer combined Z.

        Reads the integrated TFs.Pval3.txt (output of build_tfs_pval3), expands
        each peak row into one record per binding TF (a peak with non-empty
        cells in multiple TF motif columns yields multiple rows here), then
        ranks within each TF.

        Ranking score (default ``stouffer``):
            Z_atac = isf(DiffBind_FDR)
            Z_rna  = isf(RNA_padj)
            Z_comb = (Z_atac + Z_rna) / sqrt(2)
            combined_p = sf(Z_comb)
            -> sort by Z_comb desc within each TF

        Args:
            tfs_pval3_tsv:        Path to integrated TSV (build_tfs_pval3 output).
            output_tsv:           Where to write the ranked subset.
            top_k_per_tf:         Number of top targets to keep per TF (default 8).
            ranking:              "stouffer" (default), "fisher", or
                                  "effect_x_sig" (|Fold|*|log2FC|*-log10(FDR)*-log10(padj)).
            require_same_direction:
                                  Drop rows where sign(Fold) != sign(log2FC).
                                  Default True (biologically consistent TRN edges).
            fold_col, log2fc_col, fdr_col, padj_col:
                                  Column-name overrides if the source schema differs.

        Returns JSON summary with per-TF counts (before/after filter, after
        top-K), tool params, and a head preview of the ranked output.

        Output schema = input columns
            + TF             : the TF gene symbol this row's binding came from
            + score          : Stouffer Z (or whatever `ranking` computes)
            + combined_p     : sf(score) for Stouffer/Fisher
            + rank_within_tf : 1..top_k_per_tf
        """
        path = Path(tfs_pval3_tsv).expanduser()
        if not path.is_file():
            return json.dumps({"status": "error", "message": f"not found: {path}"})
        out = Path(output_tsv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(path, sep="\t")
        for c in (fold_col, log2fc_col, fdr_col, padj_col, "Gene Name"):
            if c not in df.columns:
                return json.dumps({"status": "error",
                                   "message": f"missing column '{c}'",
                                   "found":   list(df.columns)})

        # Identify HOMER motif columns: they all carry "/Homer Distance From Peak"
        # in the column header (this is the suffix HOMER annotatePeaks always
        # emits for motif occurrences). Using this pattern is robust against
        # downstream additions like combine_replicates_pval3 's "_repB" cols.
        motif_cols = [c for c in df.columns if "/Homer Distance From Peak" in c]
        if not motif_cols:
            return json.dumps({"status": "error",
                               "message": "no motif (TF) columns detected",
                               "columns": list(df.columns)})

        # Per-motif-column TF symbol (head of the column header, before '/' and '(').
        def _tf_symbol(col: str) -> str:
            head = col.split("/", 1)[0]
            m = re.match(r"\s*([A-Za-z0-9\-\.]+)", head)
            return m.group(1) if m else head.strip()
        col_to_tf = {c: _tf_symbol(c) for c in motif_cols}

        # Expand: one row per (peak, TF) where the TF cell is non-empty / not NaN.
        records: list = []
        for _, row in df.iterrows():
            for c in motif_cols:
                v = row[c]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s in ("", "nan", "NaN"):
                    continue
                rec = row.to_dict()
                rec["TF"] = col_to_tf[c]
                rec["_source_motif_col"] = c
                records.append(rec)

        if not records:
            return json.dumps({"status": "error",
                               "message": "no non-empty TF binding cells found"})

        ex = pd.DataFrame.from_records(records)
        n_expanded = len(ex)

        # Direction filter
        ex["_Fold"]   = pd.to_numeric(ex[fold_col],   errors="coerce")
        ex["_log2FC"] = pd.to_numeric(ex[log2fc_col], errors="coerce")
        ex["_FDR"]    = pd.to_numeric(ex[fdr_col],    errors="coerce")
        ex["_padj"]   = pd.to_numeric(ex[padj_col],   errors="coerce")

        n_before_dir = len(ex)
        if require_same_direction:
            ex = ex[
                ex["_Fold"].notna() & ex["_log2FC"].notna() &
                (np.sign(ex["_Fold"]) == np.sign(ex["_log2FC"])) &
                (ex["_Fold"] != 0) & (ex["_log2FC"] != 0)
            ].copy()
        n_after_dir = len(ex)

        # Ranking score
        if ranking == "stouffer":
            # isf is numerically stable for tiny p
            fdr_c   = ex["_FDR"].clip(lower=1e-300)
            padj_c  = ex["_padj"].clip(lower=1e-300)
            z_atac  = stats.norm.isf(fdr_c.to_numpy(float))
            z_rna   = stats.norm.isf(padj_c.to_numpy(float))
            ex["score"] = (z_atac + z_rna) / np.sqrt(2.0)
            ex["combined_p"] = stats.norm.sf(ex["score"].to_numpy(float))
        elif ranking == "fisher":
            fdr_c   = ex["_FDR"].clip(lower=1e-300)
            padj_c  = ex["_padj"].clip(lower=1e-300)
            chi2 = -2.0 * (np.log(fdr_c.to_numpy(float)) + np.log(padj_c.to_numpy(float)))
            ex["score"] = chi2
            ex["combined_p"] = stats.chi2.sf(chi2, df=4)
        elif ranking == "effect_x_sig":
            ex["score"] = (
                ex["_Fold"].abs()
                * ex["_log2FC"].abs()
                * -np.log10(ex["_FDR"].clip(lower=1e-300))
                * -np.log10(ex["_padj"].clip(lower=1e-300))
            )
            ex["combined_p"] = np.nan
        else:
            return json.dumps({"status": "error",
                               "message": f"unknown ranking '{ranking}'"})

        # Optional dedup: keep only the best-scoring peak per (TF, Gene Name).
        if dedup_per_tf:
            ex = ex.sort_values(["TF", "Gene Name", "score"],
                                ascending=[True, True, False], kind="mergesort")
            ex = ex.drop_duplicates(subset=["TF", "Gene Name"], keep="first")

        # Per-TF top-K
        ex = ex.sort_values(["TF", "score"], ascending=[True, False], kind="mergesort")
        ex["rank_within_tf"] = ex.groupby("TF").cumcount() + 1
        ranked = ex[ex["rank_within_tf"] <= int(top_k_per_tf)].copy()

        # Drop internal scratch columns before writing, but keep source motif column
        # (rename _source_motif_col -> source_motif_col so downstream tools, e.g.
        # AlphaGenome integration, can look up the cell value to extract the
        # motif occurrence string).
        if "_source_motif_col" in ranked.columns:
            ranked = ranked.rename(columns={"_source_motif_col": "source_motif_col"})
        for c in ["_Fold", "_log2FC", "_FDR", "_padj"]:
            if c in ranked.columns:
                ranked = ranked.drop(columns=c)

        ranked.to_csv(out, sep="\t", index=False)

        # Per-TF stats
        per_tf: dict = {}
        all_tfs = sorted(set(col_to_tf.values()))
        for tf in all_tfs:
            n_exp_tf = int((pd.DataFrame.from_records(records)["TF"] == tf).sum())
            n_dir_tf = int((ex["TF"] == tf).sum()) if not ex.empty else 0
            n_kept   = int((ranked["TF"] == tf).sum())
            per_tf[tf] = {
                "n_after_expand":     n_exp_tf,
                "n_after_dir_filter": n_dir_tf,
                "n_kept":             n_kept,
                "top_targets":        (
                    ranked[ranked["TF"] == tf]["Gene Name"].tolist()
                    if n_kept else []
                ),
            }

        return json.dumps(
            {
                "status":              "ok",
                "input":               str(path),
                "output":              str(out),
                "ranking":             ranking,
                "top_k_per_tf":        int(top_k_per_tf),
                "require_same_direction": require_same_direction,
                "dedup_per_tf":        dedup_per_tf,
                "n_rows_input":        int(len(df)),
                "n_after_expand":      int(n_expanded),
                "n_after_dir_filter":  int(n_after_dir),
                "n_final_kept":        int(len(ranked)),
                "n_tfs":               len(per_tf),
                "per_tf":              per_tf,
                "preview":             ranked.head(5).to_dict(orient="records"),
            },
            ensure_ascii=False, indent=2, default=str,
        )


    @mcp.tool()
    def combine_replicates_pval3(
        pval3_a: str,
        pval3_b: str,
        output_tsv: str,
        join_keys: Optional[List[str]] = None,
        add_rep_b_coords: bool = True,
    ) -> str:
        """Intersect two replicate TFs.Pval3.txt files by (Chr, Gene Name).

        Mirrors ``centered_peak_rev.py::new()`` -- the replicate
        aggregation step that  intended but never validated end-to-end.
        Keep only rep_A rows whose (Chr, Gene Name) pair also appears in
        rep_B. Output schema is fully compatible with rep_A
        (TFs.Pval3.txt format), so downstream rank_target_genes works
        without modification.

        Args:
            pval3_a:        Replicate A TFs.Pval3.txt (build_tfs_pval3 output).
            pval3_b:        Replicate B TFs.Pval3.txt (same condition).
            output_tsv:     Output path -- intersected, rep_A schema.
            join_keys:      Columns to join on. Default ["Chr","Gene Name"].
            add_rep_b_coords: If True, append rep_B peak coords as extra
                            columns (PeakID_B, Start_B, End_B, Annotation_B,
                            Distance to TSS_B) for downstream tracking.

        Returns JSON with row counts and per-rep stats. Condition-level
        stats columns (Fold, DiffBind_FDR, log2FC, RNA_padj) are
        replicate-invariant so they're kept from rep_A as-is.
        """
        if join_keys is None:
            join_keys = ["Chr", "Gene Name"]

        path_a = Path(pval3_a).expanduser()
        path_b = Path(pval3_b).expanduser()
        for p, lbl in ((path_a, "pval3_a"), (path_b, "pval3_b")):
            if not p.is_file():
                return json.dumps({"status": "error",
                                   "message": f"{lbl} not found: {p}"})
        out = Path(output_tsv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)

        a = pd.read_csv(path_a, sep="\t", dtype=str)
        b = pd.read_csv(path_b, sep="\t", dtype=str)
        for c in join_keys:
            if c not in a.columns:
                return json.dumps({"status": "error",
                                   "message": f"rep_A missing column '{c}'"})
            if c not in b.columns:
                return json.dumps({"status": "error",
                                   "message": f"rep_B missing column '{c}'"})

        # Set of (Chr, Gene Name) tuples in rep_B
        b_keys = set(zip(*[b[c] for c in join_keys]))
        a_keys = set(zip(*[a[c] for c in join_keys]))

        # Filter rep_A: keep only rows whose join_key tuple is in rep_B
        mask = list(zip(*[a[c] for c in join_keys]))
        a_keep = a[[k in b_keys for k in mask]].copy().reset_index(drop=True)

        if add_rep_b_coords and len(a_keep) > 0:
            # Build a lookup from rep_B: (Chr, Gene) -> first matching row's
            # position cols (PeakID, Start, End, Annotation, Distance to TSS)
            pos_cols = [c for c in ("PeakID", "Start", "End",
                                    "Annotation", "Distance to TSS")
                        if c in b.columns]
            b_first = b.drop_duplicates(subset=join_keys, keep="first")
            b_lookup = b_first.set_index(join_keys)[pos_cols]

            extras = a_keep[join_keys].apply(
                lambda r: b_lookup.loc[tuple(r[k] for k in join_keys)]
                          if tuple(r[k] for k in join_keys) in b_lookup.index
                          else pd.Series({c: None for c in pos_cols}),
                axis=1,
            )
            extras.columns = [f"{c}_repB" for c in extras.columns]
            a_keep = pd.concat([a_keep, extras.reset_index(drop=True)], axis=1)

        a_keep.to_csv(out, sep="\t", index=False)

        # Per-key statistics
        n_in_a   = len(a_keys)
        n_in_b   = len(b_keys)
        n_shared = len(a_keys & b_keys)
        n_a_only = len(a_keys - b_keys)
        n_b_only = len(b_keys - a_keys)

        return json.dumps(
            {
                "status":               "ok",
                "input_a":              str(path_a),
                "input_b":              str(path_b),
                "output":               str(out),
                "join_keys":            join_keys,
                "rep_a_rows":           int(len(a)),
                "rep_b_rows":           int(len(b)),
                "rep_a_unique_keys":    n_in_a,
                "rep_b_unique_keys":    n_in_b,
                "shared_unique_keys":   n_shared,
                "rep_a_only_keys":      n_a_only,
                "rep_b_only_keys":      n_b_only,
                "output_rows":          int(len(a_keep)),
                "fraction_a_retained":  round(len(a_keep) / max(len(a), 1), 4),
                "add_rep_b_coords":     add_rep_b_coords,
            },
            ensure_ascii=False, indent=2, default=str,
        )


    @mcp.tool()
    def join_diffbind_and_rna(
        tfs_txt: str,
        diffbind_csv: str,
        deseq_csv: str,
        output_txt: str,
        chr_col_in_tfs: str = "Chr",
        start_col_in_tfs: str = "Start",
        gene_col_in_tfs:  str = "Gene Name",
    ) -> str:
        """-exact join: TFs.txt + DiffBind nearest-peak + RNA log2FC/pval.

        Mirrors ``Step2/3.Diffbind_pval.py`` (2026-06-04). For each row
        in the filter_target_tfs output (``*.TFs.txt``):

        1. **DiffBind nearest-peak**: Look up the DiffBind region on the
           same chromosome with the smallest ``|region_start - tfs_row_start|``.
           Append its ``Fold`` and ``p.value`` as two new columns named
          ``Fold(DiffBind)`` and ``p.value(DiffBind)``.

        2. **RNA join**: For each row whose ``Gene Name`` matches a gene in
          ``DESeq2.gene.csv`` (col 1 = GENE), append ``log2FoldChange``
           (col 3 → ``log2FC``) and ``pvalue`` (col 6 → ``RNA_pval``).
           Rows whose gene is not in DESeq2 are DROPPED (reference-analysis behaviour:
           the inner-merge only writes rows where the gene matched).

        Input column index assumptions (matching the reference analysis output):
            DiffBind_out.csv: seqnames,start,end,width,strand,Conc,...,Fold,p.value,FDR
            DESeq2.gene.csv:  ID,GENE,baseMean,log2FoldChange,lfcSE,stat,pvalue,padj

        Args:
            tfs_txt:        Output of filter_target_tfs (TSV).
            diffbind_csv:   DiffBind_out.csv (the DiffBind output table).
            deseq_csv:      DESeq2.gene.csv (RNA differential expression).
            output_txt:     Destination joined TSV (.TFs.Pval2.txt).
            chr_col_in_tfs / start_col_in_tfs / gene_col_in_tfs:
                            Column names in the input TFs.txt header
                            (defaults match filter_target_tfs output).

        Returns JSON status + row counts.
        """
        # 1. Index DiffBind by chromosome
        DB: Dict[str, List[Dict]] = {}
        with open(diffbind_csv) as f:
            f.readline()  # skip header
            for line in f:
                cols = line.rstrip("\n").split(",")
                if len(cols) < 11:
                    continue
                seqnames, start, end, width, strand, \
                    Conc, ConcA, ConcB, Fold, p_value, FDR = cols[:11]
                DB.setdefault(seqnames, []).append({
                    "pos": int(float(start)),
                    "Fold": Fold,
                    "p_value": p_value,
                })

        # 2. Index DESeq2 by gene name. Mirrors 3.Diffbind_pval.py
        # DESeq2_filtered(): (a) drop rows whose pvalue is NA, (b) keep ONE
        # record per gene symbol. (a) resolves most multi-ENSG duplicates (the
        # NA isoform drops out); (b) the first surviving isoform wins for genes
        # that still map to >=2 non-NA ENSG IDs, so each gene contributes a
        # single (log2FC, RNA_pval) pair and rows never become malformed.
        RNA: Dict[str, List[Tuple[str, str]]] = {}
        with open(deseq_csv) as f:
            f.readline()  # skip header
            for line in f:
                cols = line.rstrip("\n").split(",")
                if len(cols) < 7:
                    continue
                gene = cols[1]
                fc   = cols[3]
                pval = cols[6]
                if pval == "NA":          # (a) NA pvalue -> no usable RNA signal
                    continue
                if gene in RNA:           # (b) one record per gene (first wins)
                    continue
                RNA[gene] = [(fc, pval)]

        # 3. Pass through TFs.txt: nearest-peak DiffBind + RNA join
        out_path = Path(output_txt).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_in = 0
        n_db_matched = 0
        n_rna_matched = 0
        n_emitted = 0

        with open(tfs_txt) as fp, out_path.open("w") as fpout:
            header = fp.readline().rstrip("\n")
            header_cols = header.split("\t")
            try:
                chr_idx   = header_cols.index(chr_col_in_tfs)
                start_idx = header_cols.index(start_col_in_tfs)
                gene_idx  = header_cols.index(gene_col_in_tfs)
            except ValueError as ex:
                return json.dumps({
                    "status": "error",
                    "message": f"required column missing in TFs.txt header: {ex}",
                    "header_cols": header_cols,
                })
            header_len = len(header_cols)
            # Stage 1 header (DiffBind only) and stage 2 header (full):
            fpout.write(
                header
                + "\tFold(DiffBind)\tp.value(DiffBind)"
                + "\tlog2FC\tRNA_pval\n"
            )

            for line in fp:
                n_in += 1
                line_temp = line.rstrip("\n").split("\t")
                while len(line_temp) < header_len:
                    line_temp.append("")
                Chr   = line_temp[chr_idx]
                Start = line_temp[start_idx]
                Gene  = line_temp[gene_idx]

                # DiffBind nearest-peak (reference-analysis behaviour: pure |pos - start|)
                Fold, p_value = "", ""
                nearest_distance = None
                try:
                    Start_i = int(Start)
                except (TypeError, ValueError):
                    Start_i = None
                if Start_i is not None:
                    for d in DB.get(Chr, []):
                        distance = abs(d["pos"] - Start_i)
                        if nearest_distance is None or distance < nearest_distance:
                            nearest_distance = distance
                            Fold = d["Fold"]
                            p_value = d["p_value"]
                if Fold:
                    n_db_matched += 1

                # RNA join (: inner merge — drop rows w/o RNA match)
                rna_hits = RNA.get(Gene)
                if not rna_hits:
                    continue
                n_rna_matched += 1
                # appends each (fc, pval) pair, separated by \t — for
                # genes with multiple DESeq2 rows you get extra trailing cols.
                # Practically, gene_name is unique in DESeq2.gene.csv so 1 row.
                rna_str = "\t".join("\t".join(pair) for pair in rna_hits)
                fpout.write(
                    "\t".join(line_temp)
                    + f"\t{Fold}\t{p_value}\t{rna_str}\n"
                )
                n_emitted += 1

        return json.dumps(
            {
                "status":          "ok",
                "output":          str(out_path),
                "n_rows_input":    n_in,
                "n_db_matched":    n_db_matched,
                "n_rna_matched":   n_rna_matched,
                "n_rows_emitted":  n_emitted,
            },
            ensure_ascii=False, indent=2, default=str,
        )
