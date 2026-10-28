"""Plotly interactive versions of the rank + volcano plots.

Produces Plotly JSON (interactive: hover, zoom, pan) instead of static PNG.
Matches the original R plots' color scheme (top-10 up = red, bottom-10 down = blue, rest = black)
and side-by-side layout.

ONE selection rule feeds the rank plot, the volcano and the Section-7 heatmap: up = _panel_top_genes,
down = _down_genes.  Before 2026-10-22 the rank plot and the volcano dropped every differential_MDscore < 0
row BEFORE taking their bottom 10, while the heatmap took its bottom 10 from the whole ranking, so the blue
genes of Fig 2a/2b were not the down genes of the heatmap (OC2 4 of 10 in common, PRAD 2 of 10).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional


def _normalize_score_col(df):
    """Return the canonical column name for 'Combined Score (Avg.Z)'."""
    cand = [c for c in df.columns
            if c in ("Combined Score (Avg.Z)", "Combined.Score..Avg.Z.")]
    if not cand:
        raise ValueError(
            f"INTEGRATED file missing 'Combined Score (Avg.Z)'. "
            f"Columns: {list(df.columns)}"
        )
    return cand[0]


def _panel_top_genes(df, sc, top_n, gene_col="Gene"):
    """Top-N genes for the TF-panel highlight: the Combined-Score ranking with
    negative differential_MDscore TFs dropped (control-arm-biased -> not
    treatment-program drivers), matching pick_topn / the Section-3 panel."""
    d = df.sort_values(sc, ascending=False)
    if "differential_MDscore" in d.columns:
        pos = d[d["differential_MDscore"].fillna(0) >= 0]
        if len(pos) > 0:
            d = pos
    return list(d.head(top_n)[gene_col])


def _down_genes(df, sc, top_n, gene_col="Gene"):
    """Bottom-N genes for the highlight: the N lowest Combined Scores of the WHOLE ranking.

    No differential_MDscore restriction here, unlike _panel_top_genes.  That restriction exists so the TOP
    of the ranking matches the Section-3 panel (a control-arm-biased TF must not take an activator slot).
    At the bottom end it would delete exactly the TFs whose ATAC and RNA evidence both fall -- the
    repressors the blue marks stand for -- and keep TFs whose evidence conflicts.
    Sorted by (score, Gene) so a tie at the cut-off resolves the same way whatever order the caller's
    frame is in.  Rank plot, volcano and heatmap all call this function: the blue genes of Fig 2a and 2b
    ARE the down genes of the Section-7 heatmap."""
    d = df.sort_values([sc, "Gene"], ascending=[True, True], kind="mergesort")
    return list(d.head(top_n)[gene_col])


def build_rank_plotly_json(integrated_csv: str,
                           output_json: str,
                           top_n: int = 10) -> dict:
    """Interactive rank scatter: x=rank, y=Combined Score (Avg.Z).

    Matches  rank.R: top-N red, bottom-N blue, repel labels via Plotly
    annotations.
    """
    import pandas as pd
    df = pd.read_csv(integrated_csv)
    sc = _normalize_score_col(df)
    df = df.dropna(subset=[sc, "Gene"]).copy()
    df = df.drop_duplicates(subset=["Gene"])
    # Every ranked TF is drawn and ranked (2026-10-22).  The differential_MDscore >= 0 restriction now
    # lives only in the RED selection (_panel_top_genes, matching the Section-3 panel); it used to delete
    # rows from the plot, which hid the most negative TFs and made the blue ones differ from the heatmap.
    df = df.sort_values(sc, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    n = len(df)
    _panel = _panel_top_genes(df, sc, top_n)
    _down = _down_genes(df, sc, top_n)
    df["color"] = "black"
    df.loc[df["Gene"].isin(_panel),         "color"] = "red"
    df.loc[df["Gene"].isin(_down),          "color"] = "blue"

    # 3 traces (red/black/blue) for legend hygiene
    fig = {
        "data": [],
        "layout": {
            "xaxis": {"title": "Rank"},
            "yaxis": {"title": "Combined Score (Avg.Z)",
                      "zeroline": True, "zerolinewidth": 1.5,
                      "zerolinecolor": "#888"},
            "shapes": [
                {"type":"line","yref":"y","y0":1,"y1":1,
                 "xref":"x domain","x0":0,"x1":1,
                 "line":{"color":"#999","width":1,"dash":"dash"}},
                {"type":"line","yref":"y","y0":-1,"y1":-1,
                 "xref":"x domain","x0":0,"x1":1,
                 "line":{"color":"#999","width":1,"dash":"dash"}},
            ],
            "showlegend": False,
            "margin": {"l": 60, "r": 30, "t": 30, "b": 50},
            "plot_bgcolor": "white", "paper_bgcolor": "white",
            "hovermode": "closest",
            "annotations": [],
        }
    }
    for col, label in (("red", "top"), ("black", "mid"), ("blue", "bottom")):
        sub = df[df["color"] == col]
        if not len(sub): continue
        size = sub[sc].abs()
        size_norm = 1.5 + 3 * (size / max(size.max(), 1e-9))
        fig["data"].append({
            "type": "scatter", "mode": "markers",
            "name": label,
            "x": sub["rank"].tolist(),
            "y": sub[sc].tolist(),
            "text": sub["Gene"].tolist(),
            "marker": {"color": col, "size": size_norm.tolist(),
                       "opacity": 0.9, "line": {"width": 0}},
            "hovertemplate":
                "<b>%{text}</b><br>Rank=%{x}<br>"
                "Combined Score (Avg.Z)=%{y:.2f}<extra></extra>",
        })

    # Annotations: ggrepel-style grid-fan into open whitespace (the rank.R look).
    # Red (top cluster, upper-left): labels fan up-right in a 2-col grid.
    # Blue (bottom cluster, lower-right): labels fan down-left in a 2-col grid.
    # Label anchors live in DATA coords (axref/ayref="x"/"y") so leader lines
    # reach back to each crowded marker instead of overlapping the cluster.
    top_rows = df[df["Gene"].isin(_panel)].reset_index(drop=True)
    bot_rows = df[df["Gene"].isin(_down)].reset_index(drop=True)
    y_hi = float(top_rows[sc].max()) if len(top_rows) else 1.0
    y_lo = float(bot_rows[sc].min()) if len(bot_rows) else -1.0
    rr = max((len(top_rows) + 1) // 2, 1)   # rows-per-column (red)
    br = max((len(bot_rows) + 1) // 2, 1)   # rows-per-column (blue)
    for i, (_, row) in enumerate(top_rows.iterrows()):
        col, r = i % 2, i // 2
        lx = n * (0.04 + col * 0.19)
        ly = y_hi * (1.06 - r * (0.52 / max(rr - 1, 1)))
        fig["layout"]["annotations"].append({
            "x": float(row["rank"]), "y": float(row[sc]),
            "ax": float(lx), "ay": float(ly), "axref": "x", "ayref": "y",
            "text": str(row["Gene"]),
            "showarrow": True, "arrowhead": 0, "arrowsize": 1,
            "arrowwidth": 0.6, "arrowcolor": "rgba(210,0,0,0.5)",
            "font": {"size": 11, "color": "red"},
            "xanchor": "left", "yanchor": "middle",
        })
    for i, (_, row) in enumerate(bot_rows.iterrows()):
        col, r = i % 2, i // 2
        lx = n * (0.58 + col * 0.17)
        ly = y_lo * (0.34 + r * (0.66 / max(br - 1, 1)))
        fig["layout"]["annotations"].append({
            "x": float(row["rank"]), "y": float(row[sc]),
            "ax": float(lx), "ay": float(ly), "axref": "x", "ayref": "y",
            "text": str(row["Gene"]),
            "showarrow": True, "arrowhead": 0, "arrowsize": 1,
            "arrowwidth": 0.6, "arrowcolor": "rgba(0,0,210,0.5)",
            "font": {"size": 11, "color": "blue"},
            "xanchor": "right", "yanchor": "middle",
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(fig, ensure_ascii=False))
    return {"status": "ok", "output": output_json,
            "n_points": int(n), "n_labels": int(min(top_n*2, n)),
            "up_genes": [str(g) for g in _panel], "down_genes": [str(g) for g in _down]}


def build_volcano_plotly_json(integrated_csv: str,
                              output_json: str,
                              top_n: int = 10) -> dict:
    """Interactive volcano: x=log2FC, y=Combined Score (Avg.Z).

    Matches  volcano.R: top-N red, bottom-N blue.  The highlighted genes are chosen on the SAME frame as the
    rank plot's (Gene and score only, one row per gene) by the same two functions, so Fig 2a, Fig 2b and the
    Section-7 heatmap highlight the same genes.  A selected gene with no log2FC cannot be placed on this plot;
    it is reported in the return value instead of being silently replaced by the next gene.
    """
    import pandas as pd
    df = pd.read_csv(integrated_csv)
    sc = _normalize_score_col(df)
    df = df.dropna(subset=[sc, "Gene"]).copy()
    df = df.drop_duplicates(subset=["Gene"])
    df = df.sort_values(sc, ascending=False).reset_index(drop=True)
    _panel = _panel_top_genes(df, sc, top_n)
    _down = _down_genes(df, sc, top_n)
    _placeable = df["log2FC"].notna()
    _unplaced = [str(g) for g in _panel + _down if g not in set(df.loc[_placeable, "Gene"])]
    df = df[_placeable].reset_index(drop=True)
    n = len(df)
    df["color"] = "black"
    df.loc[df["Gene"].isin(_panel),   "color"] = "red"
    df.loc[df["Gene"].isin(_down),    "color"] = "blue"

    fig = {
        "data": [],
        "layout": {
            "xaxis": {"title": "Log2 Fold Change (RNA-seq)",
                      "zeroline": True, "zerolinewidth": 1.5,
                      "zerolinecolor": "#888"},
            "yaxis": {"title": "Combined Score (Avg.Z)",
                      "zeroline": True, "zerolinewidth": 1.5,
                      "zerolinecolor": "#888"},
            "shapes": [
                {"type":"line","xref":"x","x0":0,"x1":0,
                 "yref":"y domain","y0":0,"y1":1,
                 "line":{"color":"#999","width":1,"dash":"dash"}},
                {"type":"line","yref":"y","y0":0,"y1":0,
                 "xref":"x domain","x0":0,"x1":1,
                 "line":{"color":"#999","width":1,"dash":"dash"}},
            ],
            "showlegend": False,
            "margin": {"l": 60, "r": 30, "t": 30, "b": 50},
            "plot_bgcolor": "white", "paper_bgcolor": "white",
            "hovermode": "closest",
            "annotations": [],
        }
    }
    for col in ("red", "black", "blue"):
        sub = df[df["color"] == col]
        if not len(sub): continue
        fig["data"].append({
            "type": "scatter", "mode": "markers",
            "name": col,
            "x": sub["log2FC"].tolist(),
            "y": sub[sc].tolist(),
            "text": sub["Gene"].tolist(),
            "marker": {"color": col, "size": 7, "opacity": 0.85,
                       "line": {"width": 0}},
            "hovertemplate":
                "<b>%{text}</b><br>log2FC=%{x:.2f}<br>"
                "Combined Score=%{y:.2f}<extra></extra>",
        })

    # Labels: evenly-spread leader-line annotations into open whitespace, the
    # same de-cluttering idea as Fig 2a. Red (activators) get a 2-row label
    # band ABOVE the cloud; blue (repressors) a 2-row band BELOW. Anchors are
    # spread evenly across each group's x-range so labels never overlap, and
    # leader lines reach back to each marker.
    labelled = df[df["color"] != "black"]
    x_min = float(df["log2FC"].min()); x_max = float(df["log2FC"].max())
    y_min = float(df[sc].min());       y_max = float(df[sc].max())
    xspan = max(x_max - x_min, 1e-9);  yspan = max(y_max - y_min, 1e-9)
    # Pad axes so the label bands have room
    fig["layout"]["xaxis"]["range"] = [x_min - 0.06 * xspan, x_max + 0.06 * xspan]
    fig["layout"]["yaxis"]["range"] = [y_min - 0.34 * yspan, y_max + 0.34 * yspan]

    def _band(rows, color, above):
        rows = rows.sort_values("log2FC").reset_index(drop=True)
        m = len(rows)
        if not m: return
        # Spread the label anchors across the FULL x-range rather than the
        # group's own (often very narrow) x-cluster: the top TFs all sit near
        # log2FC=0, so anchoring within their cluster piles the labels on top
        # of each other. Leader lines fan back to each marker.
        x_lo = x_min - 0.02 * xspan
        x_hi = x_max + 0.02 * xspan
        nrows = 3 if m > 6 else 2       # more vertical tiers when crowded
        for i, (_, row) in enumerate(rows.iterrows()):
            frac = i / (m - 1) if m > 1 else 0.5
            xa = x_lo + frac * (x_hi - x_lo)
            # multi-row vertical stagger to avoid neighbour collisions
            tier = i % nrows
            step = 0.06 + tier * 0.10
            ya = (y_max + step * yspan) if above else (y_min - step * yspan)
            fig["layout"]["annotations"].append({
                "x": float(row["log2FC"]), "y": float(row[sc]),
                "ax": float(xa), "ay": float(ya), "axref": "x", "ayref": "y",
                "text": str(row["Gene"]),
                "showarrow": True, "arrowhead": 0, "arrowsize": 1,
                "arrowwidth": 0.6,
                "arrowcolor": "rgba(210,0,0,0.45)" if color == "red"
                              else "rgba(0,0,210,0.45)",
                "font": {"size": 10, "color": color},
                "xanchor": "center", "yanchor": "bottom" if above else "top",
            })

    _band(labelled[labelled["color"] == "red"],  "red",  above=True)
    _band(labelled[labelled["color"] == "blue"], "blue", above=False)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(fig, ensure_ascii=False))
    return {"status": "ok", "output": output_json,
            "n_points": int(n), "n_labels": int(len(labelled)),
            "up_genes": [str(g) for g in _panel], "down_genes": [str(g) for g in _down],
            "selected_without_log2fc": _unplaced}


def register_integrated_scatter_tools(mcp):

    @mcp.tool()
    def plot_integrated_rank_interactive(
        integrated_csv: str,
        output_json: str,
        top_n: int = 10,
    ) -> str:
        """Interactive Plotly rank scatter ( rank.R equivalent)."""
        return json.dumps(
            build_rank_plotly_json(integrated_csv, output_json, top_n=top_n),
            indent=2,
        )

    @mcp.tool()
    def plot_integrated_volcano_interactive(
        integrated_csv: str,
        output_json: str,
        top_n: int = 10,
    ) -> str:
        """Interactive Plotly volcano ( volcano.R equivalent)."""
        return json.dumps(
            build_volcano_plotly_json(integrated_csv, output_json, top_n=top_n),
            indent=2,
        )


def build_heatmap_plotly_json(integrated_csv: str,
                              deseq_csv: str,
                              counts_txt: str,
                              output_json: str,
                              top_n: int = 10,
                              group_a_label: str = "OC2",
                              group_b_label: str = "Cyc",
                              samples_a: Optional[list] = None,
                              samples_b: Optional[list] = None,
                              method: str = "zscore") -> dict:
    """Interactive Plotly heatmap of top-N up + top-N down genes (RSEM counts).

    Matches Heatmap.R's scale="row" + blue-white-red palette + group
    annotation, but interactive (hover for exact Z-score).
    """
    import pandas as pd, numpy as np
    df = pd.read_csv(integrated_csv)
    sc = _normalize_score_col(df)
    df = df.dropna(subset=[sc, "Gene"]).copy()
    df = df.drop_duplicates(subset=["Gene"])    # the frame the rank plot and the volcano select on
    # The INTEGRATED Gene column is the ATAC/HOCOMOCO spelling; the DESeq2 table
    # below is in the RUN's namespace. On mouse those differ, this selection
    # matched nothing and the tool returned status "ok" with an empty heatmap
    # (2026-09-23). RNA_Gene is the expression-side spelling combine_pvalues
    # actually joined; human runs have no such column and are unchanged.
    _gene_key = "RNA_Gene" if "RNA_Gene" in df.columns else "Gene"
    top_up_genes   = _panel_top_genes(df, sc, top_n, gene_col=_gene_key)
    # Same genes as the rank plot's blue; handed over in the file's own row order, as before, so the
    # row clustering below sees the same input and an unchanged selection renders an unchanged heatmap.
    _down = set(_down_genes(df, sc, top_n, gene_col=_gene_key))
    top_down_genes = df[df[_gene_key].isin(_down)][_gene_key].tolist()
    sel_genes = [str(g) for g in top_up_genes + top_down_genes]

    rna = pd.read_csv(deseq_csv)
    # Normalise gene name column
    if "GENE" in rna.columns: gene_col = "GENE"
    elif "gene" in rna.columns: gene_col = "gene"
    elif "Gene" in rna.columns: gene_col = "Gene"
    else: gene_col = "gene_name"
    if "ID" not in rna.columns:
        rna = rna.rename(columns={"X":"ID","gene_id":"ID"})
    rna_sel = rna[rna[gene_col].isin(sel_genes)]
    if rna_sel.empty and sel_genes:
        _up = {g.upper() for g in sel_genes}
        rna_sel = rna[rna[gene_col].astype(str).str.upper().isin(_up)]
    if rna_sel.empty:
        raise ValueError(
            "no DESeq2 row matched any of the %d selected INTEGRATED genes. "
            "Joined INTEGRATED column %r (e.g. %s) against DESeq2 %r (e.g. %s) "
            "-- two different gene-symbol namespaces. Returning an empty "
            "heatmap here is what made Section 7 read as a negative result "
            "instead of a failure."
            % (len(sel_genes), _gene_key, sel_genes[:3], gene_col,
               [str(x) for x in rna[gene_col].head(3).tolist()]))
    sel_IDs = rna_sel["ID"].tolist()

    # Read counts, handle txi.rsem.counts.txt style (leading tab in header)
    with open(counts_txt) as f:
        header_line = f.readline().rstrip("\n").rstrip("\r")
    if header_line.startswith("\t"):
        counts = pd.read_csv(counts_txt, sep="\t", index_col=0)
    else:
        counts = pd.read_csv(counts_txt, sep="\t")
        # Move first column to index
        idx_col = next((c for c in counts.columns if c in ("X","gene_id","ID")), counts.columns[0])
        counts = counts.set_index(idx_col)

    counts_sub = counts.loc[counts.index.isin(sel_IDs)]
    # Replace IDs with gene symbols (preserve up-genes-first ordering)
    id_to_sym = dict(zip(rna_sel["ID"], rna_sel[gene_col]))
    counts_sub.index = [id_to_sym.get(i, i) for i in counts_sub.index]

    # Per-gene centering. Two interpretable modes:
    #   "zscore" : (x - row_mean) / row_std  -> unit-variance, classic heatmap
    #   "median" : log2(x+1) - row_median    -> normalization-based median
    #              centering (relative up/down vs the gene's typical level,
    #              no variance scaling). Robust to outlier samples.
    mat = counts_sub.values.astype(float)
    # The ROW ORDER of both panels comes from the row Z-score matrix (2026-10-22).  The median-centred panel
    # used to cluster its own matrix, so the same 20 genes stood in a different order on the left and on the
    # right of Section 7 and read as two different gene lists.  The Z-score panel is unchanged by this.
    row_mean = mat.mean(axis=1, keepdims=True)
    row_std  = mat.std(axis=1, ddof=1, keepdims=True)
    z_rows = np.where(row_std > 0, (mat - row_mean) / row_std, 0.0)
    if str(method).lower().startswith("med"):
        lm = np.log2(mat + 1.0)
        z = lm - np.median(lm, axis=1, keepdims=True)
        _cbar_title = "log2 ratio (median)"
        _hover_lbl  = "log2 vs median"
    else:
        z = z_rows
        _cbar_title = "Z-score"
        _hover_lbl  = "Z"

    samples = list(counts_sub.columns)
    half = len(samples) // 2
    arm_a_set = set(samples_a) if samples_a else set(samples[:half])
    arm_b_set = set(samples_b) if samples_b else set(samples[half:])
    # Re-order columns: arm A first, then arm B
    arm_a_cols = [c for c in samples if c in arm_a_set]
    arm_b_cols = [c for c in samples if c in arm_b_set]
    # Fallback: if named samples don't match counts columns (e.g. SRR IDs),
    # auto-split columns in half
    if not arm_a_cols and not arm_b_cols:
        half = len(samples) // 2
        arm_a_cols = samples[:half]
        arm_b_cols = samples[half:]
        # CRITICAL: refresh the membership sets too, otherwise the group-color
        # strip below ([0 if c in arm_a_set ...]) paints every column one color.
        arm_a_set = set(arm_a_cols)
        arm_b_set = set(arm_b_cols)
    col_order = arm_a_cols + arm_b_cols
    idxA = [samples.index(c) for c in arm_a_cols]
    idxB = [samples.index(c) for c in arm_b_cols]
    z_re = z[:, idxA + idxB]
    z_rows_re = z_rows[:, idxA + idxB]

    # Hierarchical row clustering (genes by Z-score profile) so co-regulated
    # genes sit together. Columns stay grouped by condition (no col cluster).
    # Always on the Z-score profile, whichever matrix is displayed, so the two panels share one row order.
    gene_labels = list(counts_sub.index)
    if z_re.shape[0] > 2:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import pdist
            _order = leaves_list(linkage(pdist(z_rows_re, metric="euclidean"),
                                         method="average"))
            z_re = z_re[_order, :]
            gene_labels = [gene_labels[k] for k in _order]
        except Exception:
            pass

    group_colors = (
        ["#d62728"] * len(arm_a_cols) +  # red  (treatment)
        ["#1f77b4"] * len(arm_b_cols)    # blue (control)
    )

    fig = {
        "data": [
            # Group annotation strip (top row)
            {
                "type": "heatmap",
                "z": [[0]*len(col_order)],   # dummy
                "x": col_order,
                "y": ["Group"],
                "showscale": False,
                "colorscale": [[0, "#d62728"], [1, "#1f77b4"]],
                "zmin": 0, "zmax": 1,
                "hoverinfo": "skip",
                "xaxis": "x", "yaxis": "y2",
            },
            # Main heatmap
            {
                "type": "heatmap",
                "z": z_re.tolist(),
                "x": col_order,
                "y": gene_labels,
                "colorscale": [
                    [0.0, "#2166AC"], [0.25, "#92C5DE"],
                    [0.5, "#FFFFFF"],
                    [0.75, "#F4A582"], [1.0, "#B2182B"],
                ],
                "zmid": 0,
                "colorbar": {
                    "title": _cbar_title, "thickness": 12, "len": 0.6,
                    "tickfont": {"size": 9},
                },
                "hovertemplate":
                    "Gene=%{y}<br>Sample=%{x}<br>"+_hover_lbl+"=%{z:.2f}<extra></extra>",
                "xaxis": "x", "yaxis": "y",
            },
        ],
        "layout": {
            "xaxis": {"title": "", "tickangle": 45, "tickfont": {"size": 10},
                      "anchor": "y"},
            "yaxis": {"title": "", "tickfont": {"size": 11},
                      "autorange": "reversed", "domain": [0.0, 0.92]},
            "yaxis2": {"domain": [0.93, 1.0], "anchor": "x",
                       "showticklabels": True, "tickfont": {"size": 9},
                       "tickvals": [], "showgrid": False},
            "annotations": [
                {"x": (len(arm_a_cols)-1)/2, "y": 1.04, "xref":"x", "yref":"paper",
                 "text": f"<b>{group_a_label}</b>", "showarrow": False,
                 "font": {"size": 11, "color": "#a01818"}},
                {"x": len(arm_a_cols) + (len(arm_b_cols)-1)/2, "y": 1.04,
                 "xref":"x", "yref":"paper",
                 "text": f"<b>{group_b_label}</b>", "showarrow": False,
                 "font": {"size": 11, "color": "#08519c"}},
            ],
            "margin": {"l": 80, "r": 60, "t": 40, "b": 60},
            "plot_bgcolor": "white", "paper_bgcolor": "white",
        },
    }

    # Color the top-strip with actual group colors per column
    fig["data"][0]["z"] = [[0 if c in arm_a_set else 1 for c in col_order]]

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(fig, ensure_ascii=False))
    return {"status": "ok", "output": output_json,
            "n_genes": int(len(counts_sub)),
            "n_samples": int(len(col_order)),
            "up_genes": [str(g) for g in top_up_genes], "down_genes": [str(g) for g in top_down_genes]}


def register_heatmap_interactive_tools(mcp):

    @mcp.tool()
    def plot_integrated_heatmap_interactive(
        integrated_csv: str,
        deseq_csv: str,
        counts_txt: str,
        output_json: str,
        top_n: int = 10,
        group_a_label: str = "OC2",
        group_b_label: str = "Cyc",
        samples_a: Optional[list] = None,
        samples_b: Optional[list] = None,
    ) -> str:
        """Interactive Plotly heatmap (Heatmap.R equivalent)."""
        return json.dumps(
            build_heatmap_plotly_json(
                integrated_csv, deseq_csv, counts_txt, output_json,
                top_n=top_n, group_a_label=group_a_label,
                group_b_label=group_b_label,
                samples_a=samples_a, samples_b=samples_b,
            ),
            indent=2,
        )
