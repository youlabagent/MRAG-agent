"""Per-TF average ATAC profile, 2x3 grid (6 TFs). One line per condition (replicate average).
Outputs Plotly JSON dict {data, layout} compatible with the report template's
Plotly.newPlot pattern (same as volcano.py / tf_rank.py)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# House convention (2026-07): treatment = RED, control = BLUE, always.
TREATMENT_COLOR = "#e31a1c"   # red
CONTROL_COLOR = "#1f78b4"     # blue

def _load_signal(bw_path: str, chrom: str, center: int, flank: int, nbins: int):
    import pyBigWig
    bw = pyBigWig.open(bw_path)
    start = max(0, center - flank); end = center + flank
    try:
        vals = bw.stats(chrom, start, end, type="mean", nBins=nbins)
    except Exception:
        vals = [None] * nbins
    finally:
        bw.close()
    return np.array([0.0 if (v is None or np.isnan(v)) else float(v) for v in vals], dtype=float)

def build_per_tf_profile_json(
    consensus_tsv: str,
    bigwig_paths: dict,           # {condition label: [bigWig, ...]}
    top_tfs: list,                # 6 TFs ordered
    flank: int = 5000,
    nbins: int = 200,
    title: str = "Per-TF average ATAC profile (OC2 vs cMyc)",
    treatment_label: str = "",    # cfg.conditions[0].label -> gets red
) -> dict:
    df = pd.read_csv(consensus_tsv, sep="\t")
    x = np.linspace(-flank/1000, flank/1000, nbins).tolist()

    # 2x3 grid axis domains
    rows, cols = 2, 3
    h_gap, v_gap = 0.07, 0.18
    panel_w = (1 - h_gap*(cols-1)) / cols
    panel_h = (1 - v_gap) / rows

    traces = []
    axes = {}
    for i, tf in enumerate(top_tfs):
        r = i // cols; c = i % cols
        xa_name = "x" if i == 0 else f"x{i+1}"
        ya_name = "y" if i == 0 else f"y{i+1}"
        # domain (Plotly uses bottom-up Y)
        xd = [c*(panel_w + h_gap), c*(panel_w + h_gap) + panel_w]
        yd_top = 1 - r*(panel_h + v_gap)
        yd_bot = yd_top - panel_h
        yd = [max(0, yd_bot), yd_top]

        axes[f"xaxis{'' if i==0 else i+1}"] = {
            "domain": xd, "anchor": ya_name,
            "title": {"text": "Distance (kb)" if r == rows-1 else "", "font":{"size":10}},
            "showgrid": False, "zeroline": False, "showline": True, "linecolor":"#999",
            "ticks": "outside", "ticklen": 3, "tickfont":{"size":9},
        }
        axes[f"yaxis{'' if i==0 else i+1}"] = {
            "domain": yd, "anchor": xa_name,
            "title": {"text": "ATAC signal" if c == 0 else "", "font":{"size":10}},
            "showgrid": False, "zeroline": False, "showline": True, "linecolor":"#999",
            "rangemode": "tozero", "ticks": "outside", "ticklen": 3, "tickfont":{"size":9},
        }

        sub = df[df["TF"].astype(str).str.upper() == tf.upper()].copy()
        if len(sub) == 0:
            continue
        sub["center"] = ((sub["Start_med"].astype(int) + sub["End_med"].astype(int)) // 2)

        cond_means = {}
        for cond, bws in bigwig_paths.items():
            mats = []
            for bw_path in bws:
                vals = np.stack([
                    _load_signal(bw_path, str(row["Chr"]), int(row["center"]), flank, nbins)
                    for _, row in sub.iterrows()
                ])
                mats.append(vals.mean(axis=0))
            cond_means[cond] = np.mean(mats, axis=0)

        # Colour follows the arm ROLE, never dict/alphabetical order: treatment
        # is always red, control always blue.  `treatment_label` is
        # cfg.conditions[0].label; when it is absent or unmatched we fall back to
        # the legacy positional mapping (first condition blue, second red).
        cond_order = list(cond_means.keys())
        _treat = str(treatment_label or "").strip().lower()

        def _is_treatment(lab: str) -> bool:
            lab = str(lab or "").strip().lower()
            if not lab or not _treat:
                return False
            return lab == _treat or lab in _treat or _treat in lab

        _role_known = any(_is_treatment(c) for c in cond_order)
        for cond, ydat in cond_means.items():
            idx = cond_order.index(cond)
            if _role_known:
                color = TREATMENT_COLOR if _is_treatment(cond) else CONTROL_COLOR
            else:
                color = CONTROL_COLOR if idx == 0 else TREATMENT_COLOR
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": x, "y": ydat.tolist(),
                "name": cond,
                "legendgroup": cond,
                "showlegend": (i == 0),
                "line": {"color": color, "width": 1.8},
                "xaxis": xa_name, "yaxis": ya_name,
                "hovertemplate": f"{tf} {cond}<br>x=%{{x:.2f}} kb<br>signal=%{{y:.2f}}<extra></extra>",
            })

    # subplot title annotations
    annotations = []
    for i, tf in enumerate(top_tfs):
        r = i // cols; c = i % cols
        x_center = c*(panel_w + h_gap) + panel_w/2
        yd_top = 1 - r*(panel_h + v_gap)
        annotations.append({
            "text": f"<b>{tf}</b>",
            "x": x_center, "y": yd_top + 0.025,
            "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 12},
            "xanchor": "center", "yanchor": "bottom",
        })

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 14}},
        "showlegend": True,
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22,
                   "yanchor": "top", "font": {"size": 11}},
        "height": 500, "margin": {"l": 60, "r": 20, "t": 60, "b": 100},
        "plot_bgcolor": "white", "paper_bgcolor": "white",
        "annotations": annotations,
        **axes,
    }
    return {"data": traces, "layout": layout, "n_tfs": len(top_tfs)}



def register_per_tf_profile_tools(mcp):
    @mcp.tool()
    def plot_per_tf_profile(
        consensus_tsv: str,
        output_json: str,
        bigwig_oc2_a: str,
        bigwig_oc2_b: str,
        bigwig_cmyc_a: str,
        bigwig_cmyc_b: str,
        top_tfs: list,
        flank: int = 5000,
        nbins: int = 200,
        title: str = "Per-TF average ATAC profile (OC2 vs cMyc)",
    ) -> dict:
        """Build Plotly JSON for per-TF average ATAC profile (2x3 grid of top TFs).

        Each panel = one TF; two lines per panel (condition averages, OC2 blue / cMyc red).
        Output JSON is consumed by render_report via per_tf_profile_json parameter.
        """
        bws = {
            "OC2":  [bigwig_oc2_a, bigwig_oc2_b],
            "cMyc": [bigwig_cmyc_a, bigwig_cmyc_b],
        }
        out = build_per_tf_profile_json(
            consensus_tsv=consensus_tsv, bigwig_paths=bws,
            top_tfs=list(top_tfs), flank=flank, nbins=nbins, title=title,
        )
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(out, indent=2))
        return {"output": output_json, "n_traces": len(out["data"]),
                "n_tfs": out.get("n_tfs", 0)}

