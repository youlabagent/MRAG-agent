"""Interactive Plotly rank + volcano nodes (Section 5)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
from ._base import Node


class InteractiveRankNode(Node):
    name = "interactive_rank_plot"

    def run(self) -> None:
        from viz_report_mcp.tools.integrated_scatter import build_rank_plotly_json
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "rank_interactive.json"
        res = build_rank_plotly_json(
            integrated_csv=self.inputs["integrated_csv"],
            output_json=str(out_json),
            top_n=self.params.get("top_n", 10),
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"interactive_rank_plot failed: {res}")
        self.outputs = {"rank_json": str(out_json)}


class InteractiveVolcanoNode(Node):
    name = "interactive_volcano_plot"

    def run(self) -> None:
        from viz_report_mcp.tools.integrated_scatter import build_volcano_plotly_json
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "volcano_interactive.json"
        res = build_volcano_plotly_json(
            integrated_csv=self.inputs["integrated_csv"],
            output_json=str(out_json),
            top_n=self.params.get("top_n", 10),
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"interactive_volcano_plot failed: {res}")
        self.outputs = {"volcano_json": str(out_json)}


class InteractiveHeatmapNode(Node):
    name = "interactive_heatmap_plot"

    def run(self) -> None:
        from viz_report_mcp.tools.integrated_scatter import build_heatmap_plotly_json
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        common = dict(
            integrated_csv=self.inputs["integrated_csv"],
            deseq_csv=self.inputs["deseq_csv"],
            counts_txt=self.inputs["counts_txt"],
            top_n=self.params.get("top_n", 10),
            group_a_label=self.params.get("group_a_label", "OC2"),
            group_b_label=self.params.get("group_b_label", "Cyc"),
            samples_a=self.inputs.get("samples_a"),
            samples_b=self.inputs.get("samples_b"),
        )
        # Right panel: classic row Z-score. Left panel: normalization-based
        # median-centering (log2 ratio vs gene median). Both produced so the
        # report can show them side by side.
        out_z = out_dir / "heatmap_interactive.json"
        rz = build_heatmap_plotly_json(output_json=str(out_z),
                                       method="zscore", **common)
        if rz.get("status") != "ok":
            raise RuntimeError(f"interactive_heatmap_plot (zscore) failed: {rz}")
        out_m = out_dir / "heatmap_median_interactive.json"
        rm = build_heatmap_plotly_json(output_json=str(out_m),
                                       method="median", **common)
        if rm.get("status") != "ok":
            raise RuntimeError(f"interactive_heatmap_plot (median) failed: {rm}")
        self.outputs = {"heatmap_json": str(out_z),
                        "heatmap_median_json": str(out_m)}
