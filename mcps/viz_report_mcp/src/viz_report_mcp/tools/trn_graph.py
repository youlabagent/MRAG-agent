"""Cytoscape.js bipartite TF->Gene graph data from ranked TFs table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _build_cy_elements(
    ranked_df: pd.DataFrame,
    *,
    tf_col: str = "TF",
    gene_col: str = "Gene Name",
    z_col: str = "Z_combined",
    atac_fc_col: str = "Fold",
    rna_fc_col: str = "RNA_log2FC",
) -> dict:
    """Generate cytoscape.js elements list (nodes + edges).

    TF nodes (square) ↔ Gene nodes (circle). Edge weight = Z, edge color = direction.
    """
    df = ranked_df.copy()
    df["abs_z"] = df[z_col].abs()
    z_max = df["abs_z"].max() if len(df) else 1.0

    # nodes — TFs
    tf_nodes = []
    for tf, g in df.groupby(tf_col):
        tf_nodes.append({
            "data": {
                "id": f"tf:{tf}",
                "label": tf,
                "is_tf": True,
                "target_count": int(len(g)),
            }
        })
    # nodes — Genes
    gene_meta = (df.groupby(gene_col)
                   .agg(z_max=("abs_z", "max"),
                        tf_count=(tf_col, "nunique"),
                        rna_fc=(rna_fc_col, "mean") if rna_fc_col in df.columns else (z_col, "mean"))
                   .reset_index())
    gene_nodes = [{
        "data": {
            "id": f"g:{r[gene_col]}",
            "label": r[gene_col],
            "is_tf": False,
            "tf_count": int(r["tf_count"]),
            "rna_fc": float(r["rna_fc"]) if not pd.isna(r["rna_fc"]) else 0.0,
        }
    } for _, r in gene_meta.iterrows()]

    # edges
    edges = []
    for i, r in df.iterrows():
        atac_fc = float(r[atac_fc_col]) if atac_fc_col in df.columns and not pd.isna(r[atac_fc_col]) else 0.0
        rna_fc  = float(r[rna_fc_col])  if rna_fc_col  in df.columns and not pd.isna(r[rna_fc_col])  else 0.0
        z = float(r[z_col])
        same_dir = (atac_fc * rna_fc) > 0 if atac_fc and rna_fc else False
        edges.append({
            "data": {
                "id": f"e:{i}",
                "source": f"tf:{r[tf_col]}",
                "target": f"g:{r[gene_col]}",
                "z": z,
                "weight": float(min(6, 1 + 4 * abs(z) / z_max if z_max else 1)),
                "atac_fc": atac_fc,
                "rna_fc": rna_fc,
                "same_dir": bool(same_dir),
                "direction": "up" if rna_fc > 0 else ("down" if rna_fc < 0 else "ns"),
            }
        })
    return {
        "elements": {"nodes": tf_nodes + gene_nodes, "edges": edges},
        "n_tf": len(tf_nodes),
        "n_gene": len(gene_nodes),
        "n_edge": len(edges),
    }


def _default_cy_style() -> list:
    return [
        {"selector": "node[is_tf]", "style": {
            "background-color": "#1f77b4",
            "shape": "round-rectangle",
            "label": "data(label)",
            "color": "#fff", "font-size": 12, "font-weight": "bold",
            "text-valign": "center", "text-halign": "center",
            "width": "label", "height": 30, "padding": 8,
            "border-width": 1, "border-color": "#0d4b7a",
        }},
        {"selector": "node[!is_tf]", "style": {
            "background-color": "#e8f0fe",
            "shape": "ellipse",
            "label": "data(label)",
            "color": "#1a3a5a", "font-size": 9,
            "text-valign": "center", "text-halign": "center",
            "width": 38, "height": 38,
            "border-width": 1, "border-color": "#9bbcdf",
        }},
        {"selector": "node[!is_tf][rna_fc > 0]",
         "style": {"background-color": "#fde0d8", "border-color": "#d6604d"}},
        {"selector": "node[!is_tf][rna_fc < 0]",
         "style": {"background-color": "#d6e8f5", "border-color": "#3182bd"}},
        {"selector": "edge", "style": {
            "curve-style": "bezier",
            "width": "data(weight)",
            "line-color": "#bbb",
            "target-arrow-color": "#bbb",
            "target-arrow-shape": "triangle",
            "opacity": 0.6,
            "arrow-scale": 0.7,
        }},
        {"selector": "edge[same_dir = 1]",
         "style": {"line-color": "#2ca02c", "target-arrow-color": "#2ca02c", "opacity": 0.85}},
        {"selector": "edge[direction = 'up']",
         "style": {"line-color": "#d62728", "target-arrow-color": "#d62728"}},
        {"selector": "edge[direction = 'down']",
         "style": {"line-color": "#1f77b4", "target-arrow-color": "#1f77b4"}},
        {"selector": "node:selected",
         "style": {"border-width": 3, "border-color": "#ff7f0e"}},
    ]


def register_trn_tools(mcp):

    @mcp.tool()
    def build_trn_elements(
        ranked_tsv: str,
        output_json: str,
        tf_col: str = "TF",
        gene_col: str = "Gene Name",
        z_col: str = "Z_combined",
        atac_fc_col: str = "Fold",
        rna_fc_col: str = "RNA_log2FC",
    ) -> dict:
        """Generate cytoscape.js elements + default style for TF->Gene network.

        Output JSON shape:
          {"elements": {"nodes": [...], "edges": [...]},
           "style": [...],
           "n_tf", "n_gene", "n_edge"}
        """
        df = pd.read_csv(ranked_tsv, sep="\t")
        cy = _build_cy_elements(df, tf_col=tf_col, gene_col=gene_col, z_col=z_col,
                                 atac_fc_col=atac_fc_col, rna_fc_col=rna_fc_col)
        out = {
            "elements": cy["elements"],
            "style": _default_cy_style(),
            "n_tf": cy["n_tf"],
            "n_gene": cy["n_gene"],
            "n_edge": cy["n_edge"],
        }
        Path(output_json).write_text(json.dumps(out))
        return {"ok": True, "output_json": output_json, "n_tf": cy["n_tf"],
                "n_gene": cy["n_gene"], "n_edge": cy["n_edge"]}
