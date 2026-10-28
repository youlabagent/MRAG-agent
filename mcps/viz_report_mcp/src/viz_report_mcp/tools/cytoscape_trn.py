"""curated-style Cytoscape inputs producer + style applier.

Generates `anno.txt` + `input_inter.txt` (same schema as the inputs of the manual
Cytoscape session test.cys) from our pipeline outputs (consensus_ranked.tsv +
INTEGRATED_ranked.csv), and produces a Cytoscape.js stylesheet that mirrors that
session's manual visual mappings exactly — including the missing-value defaults
(`NODE_FILL_COLOR=#89D0F5` (blue) for ATAC-absent,
`NODE_BORDER_PAINT=#CCCCCC` (gray) for RNA-absent).

This is the canonical, paper-ready TRN view; the mapping constants below are
decoded from `test.cys`.
"""
from __future__ import annotations
import json, string
from pathlib import Path
from typing import Optional


# Continuous mapping points (from test.cys, decoded 2026-06-07)
# Cytoscape stores [attrValue, equalValue, greaterValue, lesserValue].
# For Cytoscape.js we only need the gradient endpoints.
ATAC_FILL_STOPS = [
    (-15.0,  "#FFFFFF"),   # white ( vizmap: [-15,0] stays white; blue is only the
                           #        out-of-range <-15 floor, not an in-range anchor)
    ( -0.0356, "#F7F7F7"),  # near-white midpoint
    (+15.0,  "#FF0000"),   # red
]
RNA_BORDER_STOPS = [
    (-15.0,  "#FFF7F3"),   # pale pink low
    (  0.0,  "#F768A1"),   # pink midpoint
    (+15.0,  "#FF0000"),   # red high
]
FILL_MISSING   = "#89D0F5"   # blue  (= ATAC value absent)
BORDER_MISSING = "#CCCCCC"   # gray  (= RNA value absent)

EDGE_COLOR   = "#404040"
EDGE_WIDTH   = 2
NODE_SIZE    = 50
NODE_BORDER  = 6     # uniform;  used discrete(degree.layout)→{15,...} but
                     # for Cytoscape.js we keep this simple

GROUP_LETTERS = list(string.ascii_lowercase)   # a..z for `Interaction` column


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*[max(0, min(255, int(round(c)))) for c in rgb])


def _interpolate_color(value, stops, missing_color):
    """Piecewise-linear color interpolation through `stops` (list of (val, hex)).
    Returns the missing_color when value is None / NaN / non-numeric.
    Stops must be sorted ascending by value.
    """
    try:
        v = float(value)
        if v != v:  # NaN
            return missing_color
    except (TypeError, ValueError):
        return missing_color
    stops = sorted(stops, key=lambda s: s[0])
    if v <= stops[0][0]:
        return stops[0][1]
    if v >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i+1]
        if x0 <= v <= x1:
            t = (v - x0) / (x1 - x0) if x1 != x0 else 0.0
            rgb0 = _hex_to_rgb(c0); rgb1 = _hex_to_rgb(c1)
            return _rgb_to_hex(tuple(rgb0[k] + (rgb1[k]-rgb0[k])*t for k in range(3)))
    return missing_color


# ---------------------------------------------------------------- producer
def produce_trn_inputs(
    consensus_ranked_tsv: str,
    integrated_ranked_csv: str,
    output_dir: str,
    top_targets: int = 15,
    min_tfs_per_target: int = 2,
    pval2_txt: str = "",
    max_tfs: int = 6,
) -> dict:
    """Convert pipeline outputs into -format anno.txt + input_inter.txt.

    Args:
        consensus_ranked_tsv: per-peak ranked consensus from trn_mcp
            (cols: TF, Gene Name, Fold, log2FC, combined_p, rank_within_tf, ...)
        integrated_ranked_csv: INTEGRATED_ranked.csv (cols: Gene,
            differential_MDscore, log2FC, ..., Rank_Combined_Score)
        output_dir: directory to write anno.txt + input_inter.txt
        top_targets: number of unique target genes to include (top by combined_p)
        min_tfs_per_target: drop targets that have fewer than this many TF binders

    Returns:
        {"anno_path", "inter_path", "n_targets", "n_tfs", "n_edges"}
    """
    import pandas as pd
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    intg = pd.read_csv(integrated_ranked_csv)

    if pval2_txt:
        # --- NEW target source: treatment-arm *_peak_minmax.TFs.Pval2.txt ---
        # Select target genes by RNA log2FC (positive only), requiring the peak
        # be bound by >= min_tfs_per_target panel TFs (motif hit). This ignores
        # the DiffBind same-direction filter used by the consensus path so that
        # ATAC-down / RNA-up master targets (e.g. SRRM4) are retained.
        pv = pd.read_csv(pval2_txt, sep="\t", dtype=str)
        cols = list(pv.columns)
        gi = cols.index("Gene Name")
        fdi = cols.index("Fold(DiffBind)")
        motif_cols = cols[gi+1:fdi]
        # panel TFs = symbols the motif columns stand for (alias-aware: ANDR/ARE
        # -> AR, GCR/GRE -> NR3C1, dimers count for each member; 2026-09-10)
        import sys as _sys
        _sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
        from homer_mcp.tools.tf_alias import motif_symbols as _msym, motif_matches as _mmatch
        # only single-TF columns define the panel; dimer/composite columns
        # (ETV5::FOXO1, FOXA1:AR) still count as hits for their panel members
        # below but must not add partner TFs (FOXO1) to the max_tfs rank cut.
        panel = sorted({next(iter(_msym(c))) for c in motif_cols if len(_msym(c)) == 1})
        # keep only the top `max_tfs` regulators by INTEGRATED combined rank
        # (drops opposite-direction / low-rank panel motifs so the TRN shows
        # the same canonical TF set as the rest of the report).
        _rank_col = ("Rank_Combined_Score" if "Rank_Combined_Score" in intg.columns
                     else ("Rank" if "Rank" in intg.columns else None))
        if _rank_col is not None:
            _ranked = (intg[intg["Gene"].astype(str).isin(panel)]
                       .sort_values(_rank_col)["Gene"].astype(str).tolist())
            panel = _ranked[:max_tfs] if _ranked else panel[:max_tfs]
        else:
            panel = panel[:max_tfs]
        pv["log2FC"] = pd.to_numeric(pv["log2FC"], errors="coerce")
        pv["Fold(DiffBind)"] = pd.to_numeric(pv["Fold(DiffBind)"], errors="coerce")
        recs = []
        for gene, sub in pv.groupby("Gene Name"):
            lfc = sub["log2FC"].dropna()
            if lfc.empty:
                continue
            lfc = float(lfc.iloc[0])
            if lfc <= 0:                     # req.1: drop negative log2FC
                continue
            fold_s = sub["Fold(DiffBind)"].dropna()
            fold = float(fold_s.mean()) if not fold_s.empty else 0.0
            bound = set()
            for c in motif_cols:
                hit_tfs = [t for t in panel if _mmatch(c, t)]
                if not hit_tfs:
                    continue
                vals = sub[c].astype(str).str.strip()
                if vals.replace("nan", "").ne("").any():
                    bound.update(hit_tfs)
            if len(bound) < min_tfs_per_target:   # req.2: >=2 shared TFs
                continue
            recs.append((str(gene), lfc, fold, sorted(bound)))
        recs.sort(key=lambda r: r[1], reverse=True)   # req.2: log2FC desc
        recs = recs[:top_targets]
        selected_targets = set(r[0] for r in recs)
        edge_rows = []
        for gene, lfc, fold, bound in recs:
            for tf in bound:
                edge_rows.append({"Gene Name": gene, "TF": tf,
                                  "Fold": fold, "log2FC": lfc})
        edge_df = pd.DataFrame(edge_rows,
                               columns=["Gene Name", "TF", "Fold", "log2FC"])
        selected_tfs = set(edge_df["TF"].astype(str).tolist())
    else:
        cons = pd.read_csv(consensus_ranked_tsv, sep="\t")
        # 1. Pick top targets by best (lowest) combined_p across all TF rows
        by_target = (cons.groupby("Gene Name")
                         .agg(best_p=("combined_p", "min"),
                              n_tfs=("TF", "nunique"))
                         .reset_index())
        by_target = by_target[by_target["n_tfs"] >= min_tfs_per_target]
        by_target = by_target.sort_values("best_p").head(top_targets)
        selected_targets = set(by_target["Gene Name"].astype(str).tolist())

        # 2. Keep all TF→target rows where target in selected_targets
        edge_df = cons[cons["Gene Name"].isin(selected_targets)][
            ["Gene Name", "TF", "Fold", "log2FC"]
        ].copy()
        edge_df = edge_df.drop_duplicates(subset=["Gene Name", "TF"])
        selected_tfs = set(edge_df["TF"].astype(str).tolist())

    # 3. Build anno.txt
    # Targets: Fold = mean ATAC Fold across binding peaks, log2FC = RNA log2FC
    target_rows = []
    for gene in selected_targets:
        sub = edge_df[edge_df["Gene Name"] == gene]
        atac_fold = float(sub["Fold"].mean())
        rna_lfc   = float(sub["log2FC"].iloc[0])   # all rows share same RNA log2FC
        target_rows.append({
            "name":              gene,
            "Fold(ATAC-seq)":    atac_fold,
            "log2FC(RNA-seq)":   rna_lfc,
            "type":              "Target",
        })
    # TFs: look up own ATAC fold + RNA log2FC from INTEGRATED_ranked.csv
    intg_map = intg.set_index("Gene").to_dict("index")
    tf_rows = []
    for tf in selected_tfs:
        row = intg_map.get(tf, {})
        tf_rows.append({
            "name":              tf,
            "Fold(ATAC-seq)":    row.get("differential_MDscore", ""),
            "log2FC(RNA-seq)":   row.get("log2FC", ""),
            "type":              "TF",
        })
    anno = pd.DataFrame(target_rows + tf_rows,
                        columns=["name","Fold(ATAC-seq)","log2FC(RNA-seq)","type"])
    anno_path = out_dir / "anno.txt"
    anno.to_csv(anno_path, sep="\t", index=False, lineterminator="\r\n")

    # 4. Build input_inter.txt — Target · TF · Interaction
    # Interaction letter is per-target ( convention: each target's edges share a letter)
    target_letter = {g: GROUP_LETTERS[i % len(GROUP_LETTERS)]
                     for i, g in enumerate(sorted(selected_targets))}
    inter_rows = []
    for _, r in edge_df.iterrows():
        inter_rows.append({
            "Target":      r["Gene Name"],
            "TF":          r["TF"],
            "Interaction": target_letter[r["Gene Name"]],
        })
    inter = pd.DataFrame(inter_rows, columns=["Target","TF","Interaction"])
    inter_path = out_dir / "input_inter.txt"
    inter.to_csv(inter_path, sep="\t", index=False, lineterminator="\r\n")

    return {
        "anno_path":  str(anno_path),
        "inter_path": str(inter_path),
        "n_targets":  len(selected_targets),
        "n_tfs":      len(selected_tfs),
        "n_edges":    len(edge_df),
    }


# ---------------------------------------------------------- Cytoscape.js style
def build_trn_cyjs_style() -> list[dict]:
    """Cytoscape.js stylesheet that mirrors the manual `test.cys` style.

    Per-node fill_color and border_color are pre-computed in
    `build_trn_cyjs_elements` (Python-side gradient interpolation through the
     stops); when the source value is missing the producer assigns
    `FILL_MISSING` (blue, #89D0F5) or `BORDER_MISSING` (gray, #CCCCCC).

    Cytoscape.js then just passes them through:
    `background-color: data(fill_color)` and `border-color: data(border_color)`.
    """
    return [
        {
            "selector": "node",
            "style": {
                "label":              "data(name)",
                "color":              "#000",
                "font-size":          12,
                "font-weight":        600,
                "text-valign":        "center",
                "text-halign":        "center",
                "text-outline-color": "#fff",
                "text-outline-width": 2,
                "width":              NODE_SIZE,
                "height":             NODE_SIZE,
                "shape":              "ellipse",
                "background-color":   "data(fill_color)",
                "border-color":       "data(border_color)",
                "border-width":       NODE_BORDER,
            },
        },
        {
            "selector": 'node[type = "TF"]',
            "style": {
                "shape":  "diamond",
                "width":  NODE_SIZE + 14,
                "height": NODE_SIZE + 14,
            },
        },
        {
            "selector": "edge",
            "style": {
                "width":              EDGE_WIDTH,
                "line-color":         EDGE_COLOR,
                "target-arrow-color": EDGE_COLOR,
                "target-arrow-shape": "triangle",
                "curve-style":        "bezier",
                "opacity":            0.7,
            },
        },
    ]


# ---------------------------- Cytoscape.js elements from anno + interaction txt
def build_trn_cyjs_elements(anno_path: str, inter_path: str) -> list[dict]:
    """Read anno.txt + input_inter.txt and return Cytoscape.js elements array."""
    import pandas as pd
    anno  = pd.read_csv(anno_path, sep="\t")
    inter = pd.read_csv(inter_path, sep="\t")

    def _num(x):
        try: return float(x)
        except Exception: return None

    nodes = []
    for _, r in anno.iterrows():
        atac = _num(r.get("Fold(ATAC-seq)"))
        rna  = _num(r.get("log2FC(RNA-seq)"))
        nodes.append({
            "data": {
                "id":           str(r["name"]),
                "name":         str(r["name"]),
                "type":         str(r["type"]),
                "atac_fold":    atac,
                "rna_log2fc":   rna,
                "fill_color":   _interpolate_color(atac, ATAC_FILL_STOPS,   FILL_MISSING),
                "border_color": _interpolate_color(rna,  RNA_BORDER_STOPS, BORDER_MISSING),
            },
        })
    edges = []
    for _, r in inter.iterrows():
        edges.append({
            "data": {
                "id":          f'{r["TF"]}__to__{r["Target"]}',
                "source":      str(r["TF"]),
                "target":      str(r["Target"]),
                "interaction": str(r.get("Interaction","")),
            },
        })
    return nodes + edges


# ---------------------------- MCP registration
def register_trn_cyjs_tools(mcp):
    @mcp.tool()
    def trn_cyjs_build(
        consensus_ranked_tsv: str,
        integrated_ranked_csv: str,
        output_dir: str,
        top_targets: int = 15,
        min_tfs_per_target: int = 2,
        pval2_txt: str = "",
        max_tfs: int = 6,
    ) -> str:
        """Generate curated-style Cytoscape inputs (anno.txt + input_inter.txt) +
        Cytoscape.js elements/style JSON for the report.

        Returns JSON with paths and counts.
        """
        prod = produce_trn_inputs(
            consensus_ranked_tsv=consensus_ranked_tsv,
            integrated_ranked_csv=integrated_ranked_csv,
            output_dir=output_dir,
            top_targets=top_targets,
            min_tfs_per_target=min_tfs_per_target,
            pval2_txt=pval2_txt,
            max_tfs=max_tfs,
        )
        elements = build_trn_cyjs_elements(prod["anno_path"], prod["inter_path"])
        style    = build_trn_cyjs_style()
        cyjs_path = Path(output_dir) / "trn_cyjs.json"
        cyjs_path.write_text(json.dumps({
            "elements": elements,
            "style":    style,
            "n_nodes":  len(elements) - len([e for e in elements if "source" in e["data"]]),
            "n_edges":  len([e for e in elements if "source" in e["data"]]),
        }, indent=2))
        return json.dumps({
            "status":     "ok",
            "cyjs_json":  str(cyjs_path),
            **prod,
        }, ensure_ascii=False, indent=2)


# ======================= SE -> TF -> Target TRN (Section 10b, 2026-09-10) ===
# Extends the Section-10 graph (same Target nodes, same colour rule) with a
# super-enhancer layer: SE nodes (ROSE top-N; fill = mean DiffBind Fold of the
# peaks inside the SE, border = gray because a region has no RNA value), SE -> TF
# motif-presence edges (dashed), the enriched Section-8 motif TFs as extra
# diamonds, and each SE's ROSE CLOSEST gene appended under the Target column
# with a dotted SE -> gene edge.
SE_TF_EDGE_COLOR   = "#D08C2E"
SE_GENE_EDGE_COLOR = "#5B7DB1"
SE_NODE_W, SE_NODE_H, SE_FONT = 320, 96, 20
SE_FILL = "#FFFFFF"                        # SE boxes are always plain white (no ATAC gradient)
SE_TITLE_FONT = 40                        # "SE1" line; the coordinate line keeps SE_FONT
SE_COORD_W, SE_COORD_H, SE_PAD = 318, 86, 14   # invisible child carrying the coordinate line
TRN_NODE, TRN_TF_NODE, TRN_BORDER, TRN_FONT = 92, 112, 8, 22
SE_LAYOUT_X = {"SE": 200, "TF": 900, "T": 1600}
SE_TF_DY, SE_TF_GROUP_GAP = 140, 0.8      # TF centre spacing; extra gap (x DY) before the enriched group
SE_HEADER_DY, SE_HEADER_FONT = 190, 40    # column caption offset above the top node; caption font size
SE_HEADERS = {"SE": "Super Enhancer", "TF": "TF", "T": "Target gene"}


def _num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _read_tsv(path, delimiter="\t"):
    import csv
    if not path or not Path(path).is_file():
        return []
    with open(path, newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter=delimiter)]


def _read_atac_fallback(path: str) -> dict:
    """TF -> {differential_MDscore, ATAC_pvalue, motif_id} from either a curated
    table (header with 'Gene') or the run's raw DAStk differential file
    (no header; col0 motif id, col1 p-value, col7 differential MD score; sorted by
    p, so the first motif per symbol wins, as in combined_mcp._read_dastk)."""
    import sys as _sys
    out: dict = {}
    if not path or not Path(path).is_file():
        return out
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    if "Gene" in first.rstrip("\r\n").split("\t"):
        return {str(r.get("Gene", "")).upper(): r for r in _read_tsv(path)}
    # same id -> symbol rule as INTEGRATED.csv (combined_mcp._read_dastk), then aliases
    _sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/combined_mcp/src")
    _sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
    try:
        from combined_mcp.tools.pipeline import _hocomoco_to_symbol as _h2s
    except Exception:
        _h2s = None
    from homer_mcp.tools.tf_alias import primary_symbol as _psym0
    def _psym(short):
        if _h2s is not None:
            v = _h2s(short)
            if v != short:
                return v.upper()
        return _psym0(short)
    special = {"FOXEBOX": "FOX_EBOX"}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            mid = f[0].replace(".bed", "")
            short = mid.split("_")[0].upper()
            sym = special.get(short) or _psym(short)
            if sym and sym not in out:
                out[sym] = {"Gene": sym, "differential_MDscore": f[7], "ATAC_pvalue": f[1], "motif_id": mid}
    return out


def se_coord_node(se_el: dict) -> dict:
    """Display-only compound CHILD of an SE node carrying the coordinate line.

    Cytoscape.js has one font size per label, so "SE1" and "chr1:205632712-..."
    cannot differ in size inside one two-line label. The SE node is therefore a
    compound parent labelled "SE1" (SE_TITLE_FONT) and this child holds the
    coordinate at the original SE_FONT. Being a child it follows the parent when
    the SE node is dragged (the 2026-09-09 reason a free-standing label node was
    dropped). Never written to se_anno.txt / se_input_inter.txt."""
    d = se_el["data"]; sid = d["id"]
    return {"data": {"id": f"{sid}__coord", "parent": sid, "type": "SE_coord", "group": "SE_coord",
                     "label": d.get("region", ""), "name": d.get("name", sid),
                     "region": d.get("region", ""), "se": sid,
                     "fill_color": "#FFFFFF", "border_color": "#FFFFFF"},
            "position": dict(se_el.get("position", {"x": SE_LAYOUT_X["SE"], "y": 0})),
            "grabbable": False, "selectable": False, "classes": "se-coord"}


def apply_se_two_tier_labels(els):
    """SE label -> "SE1" only, coordinate moved to a compound child. Idempotent."""
    have = {e["data"]["id"] for e in els if e.get("data", {}).get("type") == "SE_coord"}
    out = []
    for e in els:
        d = e.get("data", {})
        if d.get("type") != "SE" or "source" in d:
            out.append(e)
            continue
        lbl = str(d.get("label", d["id"]))
        d["label"] = lbl.split(chr(10))[0].strip() or d["id"]
        out.append(e)
        if f'{d["id"]}__coord' not in have:
            out.append(se_coord_node(e))
    return out


def se_header_nodes(els: list[dict]) -> list[dict]:
    """Column captions for the Section-10 SE graph (2026-09-11, PI request).

    Display-only nodes: not grabbable, not selectable, no pointer events, never
    written to se_anno.txt / se_input_inter.txt. Positions are derived from the
    elements themselves, so re-rendering an older run gives the same captions as
    a fresh build."""
    X = SE_LAYOUT_X
    pos: dict = {}
    for e in els:
        d = e.get("data", {})
        if "source" in d or d.get("type") == "header" or "position" not in e:
            continue
        pos.setdefault(str(d.get("type", "")), []).append(float(e["position"]["y"]))
    if not pos:
        return []
    top = min(min(v) for v in pos.values()) - SE_HEADER_DY

    def _h(hid, x, y, text):
        return {"data": {"id": hid, "name": text, "label": text, "type": "header", "group": "header",
                         "fill_color": "#FFFFFF", "border_color": "#FFFFFF"},
                "position": {"x": float(x), "y": float(y)},
                "grabbable": False, "selectable": False, "classes": "hdr"}

    out = []
    if pos.get("SE"):
        out.append(_h("__hdr_SE", X["SE"], top, SE_HEADERS["SE"]))
    if pos.get("TF"):
        out.append(_h("__hdr_TF", X["TF"], top, SE_HEADERS["TF"]))
    if pos.get("Target"):
        out.append(_h("__hdr_T", X["T"], top, SE_HEADERS["T"]))
    # No caption over the SE closest-gene nodes (removed 2026-09-13, user request).
    return out


def add_tf_se_gene_edges(els: list[dict], pval2_txt: str) -> tuple[list[dict], list[dict]]:
    """TF -> SE nearest-gene edges (2026-09-14 user request).

    The nearest genes of the SEs sit under the Target column but were linked only
    to their SE. They now get TF edges by the SAME rule as the target genes: the
    TF's motif is present in a peak assigned to that gene in the extended
    (panel + enriched) Pval2 table; edge weight = motif instances. Genes without
    such a peak get no TF edge. Idempotent (existing edge ids are skipped), so the
    generator and the re-render script share it."""
    import sys as _sys
    _sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
    from homer_mcp.tools.tf_alias import motif_matches as _mmatch
    nodes = [e["data"] for e in els if "source" not in e.get("data", {})]
    tfs = [d["id"] for d in nodes if d.get("type") == "TF"]
    genes = {d["name"]: d["id"] for d in nodes if d.get("type") == "SE_gene"}
    have = {e["data"]["id"] for e in els if "source" in e.get("data", {})}
    if not tfs or not genes or not pval2_txt or not Path(pval2_txt).is_file():
        return els, []
    hit: dict = {}
    with open(pval2_txt) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        if "Gene Name" not in hdr:
            return els, []
        gi = hdr.index("Gene Name")
        fdi = hdr.index("Fold(DiffBind)") if "Fold(DiffBind)" in hdr else len(hdr)
        col_tfs = {i: [t for t in tfs if _mmatch(hdr[i], t)] for i in range(gi + 1, fdi)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= gi or f[gi] not in genes:
                continue
            for i, ts in col_tfs.items():
                if not ts:
                    continue
                cell = f[i].strip() if i < len(f) else ""
                if not cell or cell.lower() == "nan":
                    continue
                n = max(cell.count("("), 1)
                for t in ts:
                    hit[(t, f[gi])] = hit.get((t, f[gi]), 0) + n
    added = []
    for (t, g), n in sorted(hit.items()):
        eid = f"{t}__to__{genes[g]}"
        if eid in have:
            continue
        added.append({"data": {"id": eid, "source": t, "target": genes[g], "kind": "TF_SEgene",
                               "interaction": "t", "n_hits": n, "edge_source": "pval2_extended"}})
    return els + added, added


def build_se_trn_style() -> list[dict]:
    """ stylesheet (fill = ATAC Fold, border = RNA log2FC, blue/gray = missing)
    + SE / SE-linked-gene node classes + click-to-focus greying."""
    return [
        {"selector": "node", "style": {"label": "data(label)", "text-wrap": "wrap", "color": "#000",
            "font-size": TRN_FONT, "font-weight": 700, "text-valign": "center", "text-halign": "center",
            "text-outline-color": "#fff", "text-outline-width": 3, "width": TRN_NODE, "height": TRN_NODE,
            "shape": "ellipse", "background-color": "data(fill_color)", "border-color": "data(border_color)",
            "border-width": TRN_BORDER}},
        {"selector": 'node[type = "TF"]', "style": {"shape": "diamond", "width": TRN_TF_NODE, "height": TRN_TF_NODE}},
        {"selector": 'node[type = "SE"]', "style": {"shape": "round-rectangle", "font-size": SE_TITLE_FONT,
            "padding": SE_PAD, "text-valign": "top", "text-margin-y": 62, "line-height": 1.1,
            "width": SE_NODE_W, "height": SE_NODE_H,
            # compound parents are drawn below edges by default, so lines crossed the
            # SE boxes in Concentric/Circle; draw the box (and its label) on top
            "z-compound-depth": "top", "z-index": 10, "background-opacity": 1,
            "background-color": SE_FILL}},
        {"selector": 'node[type = "SE_coord"]', "style": {"shape": "rectangle", "width": SE_COORD_W,
            "height": SE_COORD_H, "background-opacity": 0, "border-width": 0, "events": "no",
            "font-size": SE_FONT, "font-weight": 600, "text-valign": "bottom", "text-margin-y": -25,
            "text-outline-width": 0, "z-compound-depth": "top", "z-index": 11}},
        {"selector": 'node[type = "SE_gene"]', "style": {"shape": "ellipse", "width": TRN_NODE, "height": TRN_NODE}},
        {"selector": 'node[type = "header"]', "style": {"shape": "rectangle", "width": 1, "height": 1,
            "background-opacity": 0, "border-width": 0, "events": "no", "font-size": SE_HEADER_FONT,
            "font-weight": 800, "color": "#1F2937", "text-outline-width": 0, "text-valign": "center",
            "text-halign": "center", "z-index": 30}},
        {"selector": "edge", "style": {"width": EDGE_WIDTH, "line-color": EDGE_COLOR, "target-arrow-color": EDGE_COLOR,
            "target-arrow-shape": "triangle", "arrow-scale": 1.2, "curve-style": "bezier", "opacity": 0.7}},
        {"selector": 'edge[kind = "SE_TF"]', "style": {"line-color": SE_TF_EDGE_COLOR, "target-arrow-color": SE_TF_EDGE_COLOR,
            "line-style": "dashed", "width": "mapData(hits_per_kb, 0, 12, 1, 6)", "opacity": 0.45}},
        {"selector": 'edge[kind = "SE_gene"]', "style": {"line-color": SE_GENE_EDGE_COLOR, "target-arrow-color": SE_GENE_EDGE_COLOR,
            "line-style": "dotted", "width": 3, "opacity": 0.8}},
        {"selector": "node.faded", "style": {"background-color": "#E6E6E6", "border-color": "#D0D0D0", "color": "#B0B0B0",
            "text-outline-width": 0, "opacity": 0.55}},
        {"selector": "edge.faded", "style": {"line-color": "#E0E0E0", "target-arrow-color": "#E0E0E0", "opacity": 0.25}},
        {"selector": "node.focus", "style": {"border-width": TRN_BORDER + 4, "z-index": 20}},
    ]


def produce_se_trn_inputs(
    graph_anno_path: str,
    graph_inter_path: str,
    se_top_tsv: str,
    se_tf_edges_tsv: str,
    se_genes_tsv: str,
    pval2_txt: str,
    integrated_csv: str,
    output_dir: str,
    panel_tfs: list[str],
    enriched_tfs: list[str],
    deseq_csv: str = "",
    atac_fallback_tsv: str = "",
    label: str = "",
) -> dict:
    """Build Section-10b inputs: se_anno.txt / se_input_inter.txt ( schema +
    SE rows) and se_cyjs.json (Cytoscape.js elements with preset 3-column layout,
    style, SE table). Target nodes are taken verbatim from the Section-10 graph
    (graph anno/inter) so both graphs show the same 20 genes; panel-TF -> Target
    edges are the Section-10 edges, enriched-TF -> Target edges come from the
    extended (panel + enriched) Pval2 table with the same promoter-presence rule.
    """
    import csv, json as _json, string as _string, sys as _sys
    _sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/homer_mcp/src")
    from homer_mcp.tools.tf_alias import motif_matches as _mmatch
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    panel_tfs = [str(t).upper() for t in (panel_tfs or [])]
    enriched_tfs = [str(t).upper() for t in (enriched_tfs or []) if str(t).upper() not in panel_tfs]
    tfs = panel_tfs + enriched_tfs
    display = {t: t.replace("FOX_EBOX", "Fox:Ebox") for t in tfs}

    # ---- Section-10 targets + panel edges (verbatim) --------------------------
    anno = _read_tsv(graph_anno_path); inter = _read_tsv(graph_inter_path)
    targets = []
    for r in anno:
        if str(r.get("type", "")).strip() == "Target":
            targets.append({"gene": r["name"].strip(), "fold": _num(r.get("Fold(ATAC-seq)")),
                            "log2fc": _num(r.get("log2FC(RNA-seq)"))})
    tset = [t["gene"] for t in targets]
    s10_edges = {(r["TF"].strip().upper(), r["Target"].strip()) for r in inter
                 if r.get("TF") and r.get("Target")}
    s10_tfs = {str(r.get("name", "")).strip().upper() for r in anno if str(r.get("type", "")).strip() == "TF"}

    # ---- extended Pval2: promoter-peak motif instances per (TF, target) --------
    hitcnt: dict = {}
    if pval2_txt and Path(pval2_txt).is_file():
        with open(pval2_txt) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            gi = hdr.index("Gene Name"); fdi = hdr.index("Fold(DiffBind)") if "Fold(DiffBind)" in hdr else len(hdr)
            mcols = [(i, hdr[i]) for i in range(gi + 1, fdi)]
            col_tfs = {i: [t for t in tfs if _mmatch(c, t)] for i, c in mcols}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) <= gi or f[gi] not in tset:
                    continue
                for i, _c in mcols:
                    cell = f[i].strip() if i < len(f) else ""
                    if not cell or cell.lower() == "nan":
                        continue
                    n = max(cell.count("("), 1)
                    for t in col_tfs[i]:
                        hitcnt[(t, f[gi])] = hitcnt.get((t, f[gi]), 0) + n
    tf_target_edges = []
    for g in tset:
        for t in tfs:
            in_s10 = (t, g) in s10_edges
            n = hitcnt.get((t, g), 0)
            # a TF drawn in the Section-10 graph keeps exactly its Section-10 edges;
            # a TF absent from it (e.g. the Fox:Ebox composite) gets promoter-presence edges
            if ((t in s10_tfs or t in panel_tfs) and in_s10) or (t not in s10_tfs and t not in panel_tfs and n > 0):
                tf_target_edges.append({"tf": t, "gene": g, "n_hits": n if n else 1,
                                        "source": "section10" if in_s10 else "pval2_extended"})

    # ---- TF attributes: INTEGRATED -> (ATAC fallback, DESeq2 fallback) ---------
    integ = {}
    for r in _read_tsv(integrated_csv, ","):
        integ[str(r.get("Gene", "")).upper()] = r
    atac_fb = _read_atac_fallback(atac_fallback_tsv)
    de = {}
    if deseq_csv and Path(deseq_csv).is_file():
        for r in _read_tsv(deseq_csv, ","):
            g = r.get("GENE") or r.get("Gene") or r.get("gene")
            if g and g.upper() not in de:
                de[g.upper()] = r
    tf_rows = []
    for t in tfs:
        i = integ.get(t, {})
        a = _num(i.get("differential_MDscore")); r_ = _num(i.get("log2FC"))
        a_src = "INTEGRATED.csv" if a is not None else ""
        r_src = "INTEGRATED.csv" if r_ is not None else ""
        if a is None and t in atac_fb:
            a = _num(atac_fb[t].get("differential_MDscore"))
            _mid = atac_fb[t].get("motif_id", "")
            _is_refill = bool(_mid) and Path("/BIO1/MRAG/Agent/Reference/ATAC_hg38/grch38_motifs_refill", _mid + ".bed").exists()
            a_src = (f"DAStk refill, display only ({_mid})" if _is_refill else f"DAStk, this run ({_mid})")
        if r_ is None and t in de:
            r_ = _num(de[t].get("log2FoldChange", de[t].get("log2FC"))); r_src = "DESeq2 (direct; TF absent from INTEGRATED)"
        tf_rows.append({"tf": t, "atac": a, "rna": r_, "atac_source": a_src, "rna_source": r_src,
                        "group": "panel" if t in panel_tfs else "enriched",
                        "rank": _num(i.get("Rank_Combined_Score"))})

    # ---- SE layer -------------------------------------------------------------
    se_rows = _read_tsv(se_top_tsv)
    se_genes = {r["SE"]: r for r in _read_tsv(se_genes_tsv)}
    se_edges = [r for r in _read_tsv(se_tf_edges_tsv) if str(r.get("present", "0")) == "1"
                and str(r.get("TF", "")).upper() in tfs]

    # ---- anno.txt / input_inter.txt ( schema + extras) -----------------------
    letters = list(_string.ascii_lowercase)
    letter = {g: letters[i % 26] for i, g in enumerate(sorted(tset))}
    anno_p = out / "se_anno.txt"; inter_p = out / "se_input_inter.txt"
    with open(anno_p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\r\n")
        w.writerow(["name", "Fold(ATAC-seq)", "log2FC(RNA-seq)", "type", "group", "rank", "region",
                    "closest_gene", "ROSE_signal", "rna_source", "atac_source"])
        for r in se_rows:
            g = se_genes.get(r["SE"], {})
            w.writerow([f'{r["SE"]} | {r["region"]}', g.get("mean_Fold_SE", ""), "", "SE", "SE", r["rank"],
                        r["region"], r.get("closest_gene", ""), r.get("signal", ""), "",
                        "DiffBind mean Fold of peaks in SE" if g.get("mean_Fold_SE") else ""])
        for t in tf_rows:
            w.writerow([t["tf"], "" if t["atac"] is None else t["atac"], "" if t["rna"] is None else t["rna"],
                        "TF", t["group"], "" if t["rank"] is None else int(t["rank"]), "", "", "",
                        t["rna_source"], t["atac_source"]])
        _tfu = {x.upper() for x in tfs}
        for i, t in enumerate(targets, 1):
            w.writerow([(t["gene"] + "__target") if t["gene"].upper() in _tfu else t["gene"],
                        "" if t["fold"] is None else t["fold"], "" if t["log2fc"] is None else t["log2fc"],
                        "Target", "Target", i, "", "", "", "Section 10 (curated)", "Section 10 (curated)"])
        _written = set(tset) | set(tfs)
        for r in se_rows:
            g = se_genes.get(r["SE"])
            if g and g.get("closest_gene") and g["closest_gene"] not in ("", "NA") and g["closest_gene"] not in _written:
                _written.add(g["closest_gene"])
                w.writerow([g["closest_gene"], g.get("mean_Fold_SE", ""), g.get("gene_log2FC", ""), "SE_gene", "SE_gene",
                            r["rank"], r["region"], "", "", "DESeq2.gene.csv", f'DiffBind mean Fold of peaks in {r["SE"]}'])
    with open(inter_p, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\r\n")
        w.writerow(["Target", "TF", "Interaction", "kind", "n_hits", "hits_per_kb"])
        for r in se_rows:
            g = se_genes.get(r["SE"])
            if g and g.get("closest_gene") and g["closest_gene"] not in ("", "NA"):
                w.writerow([g["closest_gene"], r["SE"], "g", "SE_gene", "", ""])
        for e in se_edges:
            w.writerow([e["TF"].upper(), e["SE"], "se", "SE_TF", e.get("n_hits", ""), e.get("hits_per_kb", "")])
        _tfu2 = {x.upper() for x in tfs}
        for e in tf_target_edges:
            w.writerow([(e["gene"] + "__target") if e["gene"].upper() in _tfu2 else e["gene"], e["tf"],
                        letter[e["gene"]], "TF_Target", e["n_hits"], ""])

    # ---- Cytoscape.js elements (preset 3-column layout) ---------------------
    X = SE_LAYOUT_X
    ys_tf, y = [], 0.0
    for i, t in enumerate(tf_rows):
        if i == len(panel_tfs) and enriched_tfs:
            y += SE_TF_DY * SE_TF_GROUP_GAP
        ys_tf.append(y); y += SE_TF_DY
    H = ys_tf[-1] if ys_tf else SE_TF_DY
    def spread(k): return [H * i / (k - 1) for i in range(k)] if k > 1 else [H / 2]
    els = []
    t_dy = H / (len(targets) - 1) if len(targets) > 1 else 120
    gene_y = iter([H + t_dy * 1.5 + i * t_dy for i in range(len(se_rows))])
    _tfs_up = {t.upper() for t in tfs}
    tid = {g: (f"{g}__target" if g.upper() in _tfs_up else g) for g in tset}   # a Target that is also a TF
    used_ids = set(tid.values()) | set(tfs)
    for r, yy in zip(se_rows, spread(len(se_rows))):
        g = se_genes.get(r["SE"], {}); a = _num(g.get("mean_Fold_SE"))
        els.append({"data": {"id": r["SE"], "name": f'{r["SE"]} | {r["region"]}', "type": "SE", "group": "SE",
                             "label": f'{r["SE"]}\n{r["region"]}', "region": r["region"], "closest": r.get("closest_gene", ""),
                             "rose_signal": _num(r.get("signal")), "rank": int(r["rank"]), "atac_fold": a, "rna_log2fc": None,
                             "n_peaks": _num(g.get("n_diffbind_peaks_in_SE")),
                             "atac_source": "DiffBind mean Fold of peaks in SE" if a is not None else "",
                             "fill_color": SE_FILL, "border_color": BORDER_MISSING},
                    "position": {"x": X["SE"], "y": yy}})
        if g and g.get("closest_gene") and g["closest_gene"] not in ("", "NA"):
            gname = g["closest_gene"]
            gid = tid.get(gname) or (gname.upper() if gname.upper() in _tfs_up else gname)
            els.append({"data": {"id": f'{r["SE"]}__near__{gid}', "source": r["SE"], "target": gid, "kind": "SE_gene", "interaction": "g"}})
            if gid in used_ids:      # already a Target, a TF, or another SE's nearest gene -> edge only
                continue
            used_ids.add(gid)
            rr = _num(g.get("gene_log2FC"))
            els.append({"data": {"id": gid, "name": gname, "type": "SE_gene", "group": "SE_gene", "label": gname,
                                 "se": r["SE"], "atac_fold": a, "rna_log2fc": rr, "rna_pvalue": g.get("gene_pvalue"),
                                 "atac_source": f'DiffBind mean Fold of peaks in {r["SE"]}', "rna_source": "DESeq2.gene.csv",
                                 "fill_color": _interpolate_color(a, ATAC_FILL_STOPS, FILL_MISSING),
                                 "border_color": _interpolate_color(rr, RNA_BORDER_STOPS, BORDER_MISSING)},
                        "position": {"x": X["T"], "y": next(gene_y)}})
    for t, yy in zip(tf_rows, ys_tf):
        els.append({"data": {"id": t["tf"], "name": display[t["tf"]], "type": "TF", "group": t["group"], "label": display[t["tf"]],
                             "rank": t["rank"], "atac_fold": t["atac"], "rna_log2fc": t["rna"],
                             "atac_source": t["atac_source"], "rna_source": t["rna_source"],
                             "fill_color": _interpolate_color(t["atac"], ATAC_FILL_STOPS, FILL_MISSING),
                             "border_color": _interpolate_color(t["rna"], RNA_BORDER_STOPS, BORDER_MISSING)},
                    "position": {"x": X["TF"], "y": yy}})
    for i, (t, yy) in enumerate(zip(targets, spread(len(targets))), 1):
        els.append({"data": {"id": tid[t["gene"]], "name": t["gene"], "type": "Target", "group": "Target", "label": t["gene"],
                             "rank": i, "atac_fold": t["fold"], "rna_log2fc": t["log2fc"],
                             "atac_source": "Section 10", "rna_source": "Section 10",
                             "fill_color": _interpolate_color(t["fold"], ATAC_FILL_STOPS, FILL_MISSING),
                             "border_color": _interpolate_color(t["log2fc"], RNA_BORDER_STOPS, BORDER_MISSING)},
                    "position": {"x": X["T"], "y": yy}})
    for e in se_edges:
        els.append({"data": {"id": f'{e["SE"]}__to__{e["TF"].upper()}', "source": e["SE"], "target": e["TF"].upper(),
                             "kind": "SE_TF", "interaction": "se", "n_hits": int(float(e.get("n_hits") or 0)),
                             "hits_per_kb": float(e.get("hits_per_kb") or 0.0)}})
    for e in tf_target_edges:
        els.append({"data": {"id": f'{e["tf"]}__to__{tid[e["gene"]]}', "source": e["tf"], "target": tid[e["gene"]], "kind": "TF_Target",
                             "interaction": letter[e["gene"]], "n_hits": e["n_hits"], "edge_source": e["source"]}})

    els, tf_gene_edges = add_tf_se_gene_edges(els, pval2_txt)   # TF -> SE nearest gene
    if tf_gene_edges:
        with open(inter_p, "a", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\r\n")
            for e in tf_gene_edges:
                d = e["data"]
                w.writerow([d["target"], d["source"], "t", "TF_SEgene", d["n_hits"], ""])
    # cytoscape.js adds elements in order: make sure every edge follows its endpoint nodes
    els = [e for e in els if "source" not in e["data"]] + [e for e in els if "source" in e["data"]]
    els = apply_se_two_tier_labels(els)         # big "SE1" + coordinate child
    els = se_header_nodes(els) + els            # column captions (display only)
    se_table = []
    for r in se_rows:
        g = se_genes.get(r["SE"], {})
        se_table.append({"se": r["SE"], "region": r["region"], "closest_gene": r.get("closest_gene", ""),
                         "signal": _num(r.get("signal")), "n_tf": sum(1 for e in se_edges if e["SE"] == r["SE"]),
                         "n_peaks": g.get("n_diffbind_peaks_in_SE", ""), "mean_fold": g.get("mean_Fold_SE", ""),
                         "gene_log2fc": _num(g.get("gene_log2FC")), "gene_pvalue": _num(g.get("gene_pvalue"))})
    fb_atac = [display[t["tf"]] for t in tf_rows if t["atac_source"].startswith("DAStk")]
    fb_rna = [display[t["tf"]] for t in tf_rows if t["rna_source"].startswith("DESeq2")]
    miss_atac = [display[t["tf"]] for t in tf_rows if t["atac"] is None]
    miss_rna = [display[t["tf"]] for t in tf_rows if t["rna"] is None]
    summary = {"label": label, "n_se": len(se_rows), "n_tf": len(tfs), "n_panel": len(panel_tfs),
               "n_enriched": len(enriched_tfs), "n_targets": len(targets), "n_se_genes": len([s for s in se_table if s["closest_gene"]]),
               "n_se_tf_edges": len(se_edges), "n_tf_target_edges": len(tf_target_edges),
               "n_tf_se_gene_edges": len(tf_gene_edges),
               "n_se_genes_with_tf": len({e["data"]["target"] for e in tf_gene_edges}),
               "n_tf_target_edges_section10": sum(1 for e in tf_target_edges if e["source"] == "section10"),
               "panel_tfs": panel_tfs, "enriched_tfs": [display[t] for t in enriched_tfs],
               "tf_atac_fallback": fb_atac, "tf_rna_fallback": fb_rna, "tf_atac_missing": miss_atac, "tf_rna_missing": miss_rna}
    cyjs = {"elements": els, "style": build_se_trn_style(), "layout": "preset", "se_table": se_table, "summary": summary,
            "colour_stops": {"ATAC_FILL_STOPS": ATAC_FILL_STOPS, "RNA_BORDER_STOPS": RNA_BORDER_STOPS,
                             "FILL_MISSING": FILL_MISSING, "BORDER_MISSING": BORDER_MISSING}}
    cy_p = out / "se_cyjs.json"; cy_p.write_text(_json.dumps(cyjs, indent=1))
    return {"se_cyjs_json": str(cy_p), "se_anno_path": str(anno_p), "se_inter_path": str(inter_p), **summary}
