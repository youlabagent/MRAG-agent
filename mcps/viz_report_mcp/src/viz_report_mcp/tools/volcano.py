"""Interactive Plotly volcano plots for ATAC (DiffBind) and RNA (DESeq2)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _nearest_genes(peak_ids, gtf_path: str, window: int = 200000) -> dict:
    """Map peak-coordinate ids ("chr:start-end") to the nearest gene_name using
    a tabix-indexed GENCODE GTF. Overlapping gene wins (distance 0); otherwise
    the closest gene body within +/-window. Falls back to the peak id when no
    gene is found or on any error, so labels never go blank."""
    import subprocess, re
    out = {}
    rx = re.compile(r'gene_name "([^"]+)"')
    for pid in peak_ids:
        out[pid] = pid
        try:
            chrom, rng = str(pid).split(":")
            a, b = rng.split("-"); s = int(a); e = int(b)
        except Exception:
            continue
        mid = (s + e) // 2
        lo = max(0, mid - window); hi = mid + window
        try:
            res = subprocess.run(["tabix", gtf_path, f"{chrom}:{lo}-{hi}"],
                                 capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        best = None; bestd = None
        for line in res.stdout.splitlines():
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            gs = int(f[3]); ge = int(f[4])
            if   e < gs: dist = gs - e
            elif s > ge: dist = s - ge
            else:        dist = 0
            if bestd is None or dist < bestd:
                m = rx.search(f[8])
                if m:
                    best = m.group(1); bestd = dist
                    if dist == 0:
                        break
        if best:
            out[pid] = best
    return out


def _build_volcano_json(
    df: pd.DataFrame,
    *,
    gene_col: str,
    fc_col: str,
    pval_col: str,
    fdr_col: Optional[str],
    title: str,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    top_label_n: int = 20,
    slim: bool = False,
    slim_top_per_side: int = 2500,
    slim_ns_sample: int = 500,
    gtf_path: Optional[str] = None,
) -> dict:
    """Build Plotly JSON for volcano. slim=True dramatically reduces point count.

    slim mode:
      - keep top N significant up (by |log2FC|*nlog10p)
      - keep top N significant down (likewise)
      - random sample of NS points for context
      Reduces 183k+ points to ~5k. File size drops ~30x.
    """
    d = df[[gene_col, fc_col, pval_col] + ([fdr_col] if fdr_col and fdr_col in df else [])].copy()
    d = d.rename(columns={gene_col: "gene", fc_col: "log2FC", pval_col: "pvalue"})
    d["pvalue"] = d["pvalue"].clip(lower=1e-300)
    d["nlog10p"] = -np.log10(d["pvalue"])
    if fdr_col and fdr_col in df:
        d["FDR"] = df[fdr_col].clip(lower=1e-300)
    d = d.dropna(subset=["log2FC", "pvalue"])

    sig_up = (d["log2FC"] >= fc_threshold) & (d["pvalue"] <= pval_threshold)
    sig_dn = (d["log2FC"] <= -fc_threshold) & (d["pvalue"] <= pval_threshold)
    d["cat"] = "ns"
    d.loc[sig_up, "cat"] = "up"
    d.loc[sig_dn, "cat"] = "down"
    d["score"] = d["log2FC"].abs() * d["nlog10p"]

    n_total = len(d); n_up = int(sig_up.sum()); n_down = int(sig_dn.sum())

    if slim:
        up_sub   = d[d["cat"] == "up"].nlargest(slim_top_per_side, "score")
        down_sub = d[d["cat"] == "down"].nlargest(slim_top_per_side, "score")
        ns_sub   = d[d["cat"] == "ns"].sample(min(slim_ns_sample, int((d["cat"] == "ns").sum())), random_state=0)
        d = pd.concat([up_sub, down_sub, ns_sub], ignore_index=True)

    top = d.sort_values("score", ascending=False).head(top_label_n)["gene"].tolist()

    def trace(cat, color, name):
        sub = d[d["cat"] == cat]
        hover = [f"{g}<br>log2FC: {fc:.2f}<br>p: {p:.2e}"
                 for g, fc, p in zip(sub["gene"], sub["log2FC"], sub["pvalue"])]
        return {
            "x": sub["log2FC"].tolist(),
            "y": sub["nlog10p"].tolist(),
            "text": sub["gene"].tolist(),
            "hovertext": hover,
            "mode": "markers",
            "type": "scattergl",
            "name": name,
            "marker": {"color": color, "size": 4, "opacity": 0.7},
            "hoverinfo": "text",
        }

    traces = [
        trace("ns",   "#cccccc", "ns"),
        trace("up",   "#d62728", "up"),
        trace("down", "#1f77b4", "down"),
    ]
    label_d = d[d["gene"].isin(top)]
    _label_text = label_d["gene"].tolist()
    if gtf_path:
        _gmap = _nearest_genes(label_d["gene"].tolist(), gtf_path)
        _label_text = [_gmap.get(g, g) for g in label_d["gene"].tolist()]
    traces.append({
        "x": label_d["log2FC"].tolist(),
        "y": label_d["nlog10p"].tolist(),
        "text": _label_text,
        "mode": "text",
        "type": "scatter",
        "name": "labels",
        "textposition": "top center",
        "textfont": {"size": 9, "color": "#222"},
        "showlegend": False,
        "hoverinfo": "skip",
    })
    layout = {
        "title": {"text": title + (f" (slim: {len(d)} of {n_total})" if slim else ""), "x": 0.5, "font": {"size": 13}},
        "xaxis": {"title": "log2 fold change", "zeroline": True, "zerolinecolor": "#888"},
        "yaxis": {"title": "-log10(p-value)", "zeroline": False},
        "hovermode": "closest",
        "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
        "legend": {"x": 1.02, "y": 1, "font": {"size": 10}},
        "shapes": [
            {"type": "line", "x0": fc_threshold, "x1": fc_threshold,
             "y0": 0, "y1": d["nlog10p"].max(), "line": {"dash": "dot", "color": "#888", "width": 1}},
            {"type": "line", "x0": -fc_threshold, "x1": -fc_threshold,
             "y0": 0, "y1": d["nlog10p"].max(), "line": {"dash": "dot", "color": "#888", "width": 1}},
            {"type": "line", "x0": d["log2FC"].min(), "x1": d["log2FC"].max(),
             "y0": -math.log10(pval_threshold), "y1": -math.log10(pval_threshold),
             "line": {"dash": "dot", "color": "#888", "width": 1}},
        ],
    }
    return {"data": traces, "layout": layout, "n_total": n_total, "n_up": n_up, "n_down": n_down, "n_displayed": int(len(d))}





def register_volcano_tools(mcp):

    @mcp.tool()
    def plot_volcano(
        input_csv: str,
        output_json: str,
        gene_col: str,
        fc_col: str,
        pval_col: str,
        fdr_col: Optional[str] = None,
        title: str = "Volcano",
        fc_threshold: float = 1.0,
        pval_threshold: float = 0.05,
        top_label_n: int = 20,
        slim: bool = False,
        slim_top_per_side: int = 2500,
        slim_ns_sample: int = 500,
    ) -> dict:
        """Build a Plotly volcano JSON from a differential CSV (ATAC or RNA).

        Args:
            input_csv: DiffBind_out.csv or DESeq2.gene.csv
            output_json: where to save Plotly JSON (data+layout)
            gene_col, fc_col, pval_col, fdr_col: column names in input
            title, fc_threshold, pval_threshold, top_label_n: rendering
        Returns: stats dict (n_total, n_up, n_down, output path)
        """
        df = pd.read_csv(input_csv)
        fig = _build_volcano_json(
            df, gene_col=gene_col, fc_col=fc_col, pval_col=pval_col,
            fdr_col=fdr_col, title=title,
            fc_threshold=fc_threshold, pval_threshold=pval_threshold,
            top_label_n=top_label_n,
            slim=slim, slim_top_per_side=slim_top_per_side, slim_ns_sample=slim_ns_sample,
        )
        Path(output_json).write_text(json.dumps(fig))
        return {
            "ok": True,
            "output_json": output_json,
            "n_total": fig["n_total"],
            "n_up": fig["n_up"],
            "n_down": fig["n_down"],
            "n_displayed": fig["n_displayed"],
        }
