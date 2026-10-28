"""TF combined-score x log2FC scatter (poster style).

Replaces the rank-only plot with a 2D view: x = TF's RNA log2FC,
y = combined Avg.Z (sign-aware ATAC+RNA Stouffer). The four quadrants then
have biological meaning:
  - top-right    (high Z, FC > 0): activator + own expression up
  - bottom-left  (low Z,  FC < 0): repressor + own expression down
  - top-left / bottom-right: inconsistent
Threshold dashed lines + green highlight of consistent TFs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


def _compute_avg_z(df: pd.DataFrame, atac_p_col: str = "ATAC_pvalue",
                    rna_p_col: str = "RNA_pvalue",
                    md_col: str = "differential_MDscore",
                    log2fc_col: str = "log2FC",
                    clip: float = 1e-30) -> np.ndarray:
    """Signed Stouffer-style Avg.Z. ATAC sign from MD-score, RNA sign from log2FC."""
    z_a = norm.isf(np.clip(df[atac_p_col].values, clip, 1.0)) * np.sign(df[md_col].values)
    z_r = norm.isf(np.clip(df[rna_p_col].values, clip, 1.0)) * np.sign(df[log2fc_col].values)
    return (z_a + z_r) / 2.0


def _build_fc_vs_z_json(
    integrated_df: pd.DataFrame,
    *,
    gene_col: str = "Gene",
    log2fc_col: str = "log2FC",
    atac_p_col: str = "ATAC_pvalue",
    rna_p_col: str = "RNA_pvalue",
    md_col: str = "differential_MDscore",
    pcom_col: str = "Pcom",
    fc_threshold: float = 1.0,
    z_threshold: float = 2.0,
    label_top_n: int = 10,
    label_bottom_n: int = 7,
    title: str = "TF combined evidence vs log2 fold change",
    clip: float = 1e-30,
) -> dict:
    df = integrated_df.copy()
    df = df.dropna(subset=[log2fc_col, atac_p_col, rna_p_col, md_col])
    df["avg_z"] = _compute_avg_z(df, atac_p_col=atac_p_col, rna_p_col=rna_p_col,
                                   md_col=md_col, log2fc_col=log2fc_col, clip=clip)
    df = df.reset_index(drop=True)
    df["rank"] = df["avg_z"].rank(ascending=False, method="first").astype(int)

    # Consistent quadrants
    cons_pos = (df[log2fc_col] >  fc_threshold) & (df["avg_z"] >  z_threshold)
    cons_neg = (df[log2fc_col] < -fc_threshold) & (df["avg_z"] < -z_threshold)
    df["consistent"] = cons_pos | cons_neg

    # Pick labeled subset: top N consistent positive + bottom N consistent negative
    top   = df[cons_pos].sort_values("avg_z", ascending=False).head(label_top_n)
    bot   = df[cons_neg].sort_values("avg_z", ascending=True ).head(label_bottom_n)
    labeled = pd.concat([top, bot])
    labeled_genes = set(labeled[gene_col])

    def trace_cat(mask, color, name, size, line_color=None):
        sub = df[mask]
        hov = [f"<b>{r[gene_col]}</b><br>log2FC={r[log2fc_col]:.2f}<br>Avg.Z={r['avg_z']:.2f}<br>Pcom={r[pcom_col]:.2e}<br>rank={r['rank']}"
               for _, r in sub.iterrows()]
        m = {"color": color, "size": size, "opacity": 0.85}
        if line_color:
            m["line"] = {"color": line_color, "width": 1}
        return {
            "x": sub[log2fc_col].tolist(),
            "y": sub["avg_z"].tolist(),
            "hovertext": hov,
            "mode": "markers",
            "type": "scatter",
            "name": name,
            "marker": m,
            "hoverinfo": "text",
        }

    # Background (gray)
    bg_mask = ~df["consistent"]
    traces = [
        trace_cat(bg_mask, "#bbbbbb", "other", 5),
        trace_cat(cons_pos, "#2ca02c", "consistent ↑ (activator)", 8, line_color="#1f7a1f"),
        trace_cat(cons_neg, "#2ca02c", "consistent ↓ (repressor)", 8, line_color="#1f7a1f"),
    ]
    # Text labels for top + bottom
    top_lab = top
    bot_lab = bot
    if len(top_lab):
        traces.append({
            "x": top_lab[log2fc_col].tolist(),
            "y": top_lab["avg_z"].tolist(),
            "text": top_lab[gene_col].tolist(),
            "mode": "text",
            "type": "scatter",
            "name": "labels_up",
            "textposition": "top right",
            "textfont": {"size": 11, "color": "#d62728"},
            "showlegend": False,
            "hoverinfo": "skip",
        })
    if len(bot_lab):
        traces.append({
            "x": bot_lab[log2fc_col].tolist(),
            "y": bot_lab["avg_z"].tolist(),
            "text": bot_lab[gene_col].tolist(),
            "mode": "text",
            "type": "scatter",
            "name": "labels_dn",
            "textposition": "bottom left",
            "textfont": {"size": 11, "color": "#1f77b4"},
            "showlegend": False,
            "hoverinfo": "skip",
        })

    x_min, x_max = float(df[log2fc_col].min()), float(df[log2fc_col].max())
    y_min, y_max = float(df["avg_z"].min()),    float(df["avg_z"].max())
    pad_x = max(1, 0.05 * (x_max - x_min))
    pad_y = max(0.5, 0.05 * (y_max - y_min))

    layout = {
        "title": {"text": title, "x": 0.5, "font": {"size": 13}},
        "xaxis": {"title": "Log2 Fold Change (RNA-seq)", "zeroline": False, "range": [x_min - pad_x, x_max + pad_x]},
        "yaxis": {"title": "Combined Score (Avg.Z)", "zeroline": False, "range": [y_min - pad_y, y_max + pad_y]},
        "hovermode": "closest",
        "margin": {"t": 40, "r": 30, "b": 50, "l": 60},
        "legend": {"x": 1.02, "y": 1, "font": {"size": 10}},
        "shapes": [
            # vertical FC thresholds
            {"type":"line","x0":fc_threshold,"x1":fc_threshold,
             "y0":y_min-pad_y,"y1":y_max+pad_y,
             "line":{"dash":"dash","color":"#666","width":1}},
            {"type":"line","x0":-fc_threshold,"x1":-fc_threshold,
             "y0":y_min-pad_y,"y1":y_max+pad_y,
             "line":{"dash":"dash","color":"#666","width":1}},
            # horizontal Z thresholds
            {"type":"line","x0":x_min-pad_x,"x1":x_max+pad_x,
             "y0":z_threshold,"y1":z_threshold,
             "line":{"dash":"dot","color":"#666","width":1}},
            {"type":"line","x0":x_min-pad_x,"x1":x_max+pad_x,
             "y0":-z_threshold,"y1":-z_threshold,
             "line":{"dash":"dot","color":"#666","width":1}},
            # zero axes
            {"type":"line","x0":0,"x1":0,"y0":y_min-pad_y,"y1":y_max+pad_y,
             "line":{"dash":"solid","color":"#dddddd","width":1}},
            {"type":"line","x0":x_min-pad_x,"x1":x_max+pad_x,"y0":0,"y1":0,
             "line":{"dash":"solid","color":"#dddddd","width":1}},
        ],
    }
    return {
        "data": traces,
        "layout": layout,
        "n_total": int(len(df)),
        "n_consistent_up": int(cons_pos.sum()),
        "n_consistent_down": int(cons_neg.sum()),
        "n_labeled": int(len(labeled_genes)),
    }


def register_tf_rank_tools(mcp):

    @mcp.tool()
    def plot_tf_rank(
        integrated_csv: str,
        output_json: str,
        gene_col: str = "Gene",
        log2fc_col: str = "log2FC",
        atac_p_col: str = "ATAC_pvalue",
        rna_p_col: str = "RNA_pvalue",
        md_col: str = "differential_MDscore",
        pcom_col: str = "Pcom",
        fc_threshold: float = 1.0,
        z_threshold: float = 2.0,
        label_top_n: int = 10,
        label_bottom_n: int = 7,
        title: str = "TF combined evidence vs log2 fold change",
    ) -> dict:
        """Build TF log2FC × Avg.Z scatter (poster style).

        Quadrants:
          top-right (consistent ↑): activator + own RNA up
          bottom-left (consistent ↓): repressor + own RNA down
          off-quadrants: ATAC ↔ RNA inconsistent
        """
        df = pd.read_csv(integrated_csv)
        fig = _build_fc_vs_z_json(
            df, gene_col=gene_col, log2fc_col=log2fc_col,
            atac_p_col=atac_p_col, rna_p_col=rna_p_col, md_col=md_col,
            pcom_col=pcom_col, fc_threshold=fc_threshold, z_threshold=z_threshold,
            label_top_n=label_top_n, label_bottom_n=label_bottom_n, title=title,
        )
        Path(output_json).write_text(json.dumps(fig))
        return {
            "ok": True,
            "output_json": output_json,
            "n_total": fig["n_total"],
            "n_consistent_up": fig["n_consistent_up"],
            "n_consistent_down": fig["n_consistent_down"],
            "n_labeled": fig["n_labeled"],
        }
