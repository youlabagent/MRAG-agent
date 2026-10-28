"""Standalone wrapper for rank_target_genes logic (callable without FastMCP).

Mirrors the body of trn_mcp.tools.integrate.register_integrate_tools.rank_target_genes
so orchestrator code can call it as a plain Python function.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def rank_target_genes(
    tfs_pval3_tsv: str,
    output_tsv: str,
    top_k_per_tf: int = 8,
    ranking: str = "stouffer",
    require_same_direction: bool = True,
    balance_direction: bool = False,
    dedup_per_tf: bool = True,
    fold_col: str = "Fold",
    log2fc_col: str = "log2FC",
    fdr_col: str = "DiffBind_FDR",
    padj_col: str = "RNA_padj",
) -> str:
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

    motif_cols = [c for c in df.columns if "/Homer Distance From Peak" in c]
    if not motif_cols:
        return json.dumps({"status": "error",
                           "message": "no motif (TF) columns detected",
                           "columns": list(df.columns)})

    def _tf_symbol(col: str) -> str:
        # Try to extract from the ChIP-Seq experiment label (second "/" segment).
        # Falls back to the motif alias (first segment).
        # Always returns UPPERCASE to dedupe HOCOMOCOv12 vs JASPAR2024 case
        # variants (GATA3 vs Gata3, STAT6 vs Stat6, etc.).
        parts = col.split("/")
        if len(parts) >= 2:
            exp = parts[1]
            m = re.search(r"-([A-Za-z][A-Za-z0-9]+[0-9A-Za-z]*)-ChIP-Seq", exp)
            if m:
                return m.group(1).upper()
        head = parts[0]
        m = re.match(r"\s*([A-Za-z0-9\-\.]+)", head)
        out = m.group(1) if m else head.strip()
        return out.upper()
    col_to_tf = {c: _tf_symbol(c) for c in motif_cols}

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
    ex["_Fold"]   = pd.to_numeric(ex[fold_col],   errors="coerce")
    ex["_log2FC"] = pd.to_numeric(ex[log2fc_col], errors="coerce")
    ex["_FDR"]    = pd.to_numeric(ex[fdr_col],    errors="coerce")
    ex["_padj"]   = pd.to_numeric(ex[padj_col],   errors="coerce")

    if require_same_direction:
        ex = ex[
            ex["_Fold"].notna() & ex["_log2FC"].notna() &
            (np.sign(ex["_Fold"]) == np.sign(ex["_log2FC"])) &
            (ex["_Fold"] != 0) & (ex["_log2FC"] != 0)
        ].copy()

    if ranking == "stouffer":
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

    if dedup_per_tf:
        ex = ex.sort_values(["TF", "Gene Name", "score"],
                            ascending=[True, True, False], kind="mergesort")
        ex = ex.drop_duplicates(subset=["TF", "Gene Name"], keep="first")

    ex = ex.sort_values(["TF", "score"], ascending=[True, False], kind="mergesort")
    if balance_direction:
        # Per TF, keep top_k targets split ~50/50 between the two accessibility
        # directions (Fold>0 gained in arm_a vs Fold<0 gained in arm_b), instead
        # of taking the top-k by score regardless of direction. Otherwise the
        # strongest concordant targets can all fall on one side, hiding the
        # other condition's gained loci in the downstream locus panels.
        k = int(top_k_per_tf)
        k_up = (k + 1) // 2
        k_dn = k - k_up
        parts = []
        for _tf, g in ex.groupby("TF", sort=False):
            up = g[g["_Fold"] > 0].head(k_up)
            dn = g[g["_Fold"] < 0].head(k_dn)
            short = k - len(up) - len(dn)
            if short > 0:
                _used = set(up.index) | set(dn.index)
                extra = g[~g.index.isin(_used)].head(short)
                sel = pd.concat([up, dn, extra])
            else:
                sel = pd.concat([up, dn])
            parts.append(sel.sort_values("score", ascending=False, kind="mergesort"))
        ranked = pd.concat(parts).copy() if parts else ex.head(0).copy()
        ranked["rank_within_tf"] = ranked.groupby("TF").cumcount() + 1
    else:
        ex["rank_within_tf"] = ex.groupby("TF").cumcount() + 1
        ranked = ex[ex["rank_within_tf"] <= int(top_k_per_tf)].copy()

    if "_source_motif_col" in ranked.columns:
        ranked = ranked.rename(columns={"_source_motif_col": "source_motif_col"})
    for c in ["_Fold", "_log2FC", "_FDR", "_padj"]:
        if c in ranked.columns:
            ranked = ranked.drop(columns=c)
    ranked.to_csv(out, sep="\t", index=False)

    return json.dumps({
        "status": "ok",
        "input": str(path),
        "output": str(out),
        "n_rows_input": int(len(df)),
        "n_expanded": int(n_expanded),
        "n_final_kept": int(len(ranked)),
        "n_tfs": int(ranked["TF"].nunique()) if "TF" in ranked.columns else 0,
    })
