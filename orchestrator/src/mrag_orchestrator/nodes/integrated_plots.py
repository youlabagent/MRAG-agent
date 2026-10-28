""" R-plot orchestrator nodes (rank / volcano / heatmap)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/BIO1/MRAG/Agent/mcps/viz_report_mcp/src")
from ._base import Node, capture_mcp_tools


class RankPlotNode(Node):
    """ rank.R wrap → rank vs Combined Score scatter PNG."""
    name = "rank_plot"

    def run(self) -> None:
        from viz_report_mcp.tools.integrated_plots import register_integrated_plot_tools
        tools = capture_mcp_tools(register_integrated_plot_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "rank.png"
        res = json.loads(tools["plot_rank"](
            integrated_csv=self.inputs["integrated_csv"],
            output_png=str(out_png),
            top_n=self.params.get("top_n", 10),
            label_size=self.params.get("label_size", 3.0),
        ))
        if res.get("status") != "ok":
            raise RuntimeError(f"plot_rank failed: {res.get('stderr','')[:300]}")
        self.outputs = {"rank_png": str(out_png)}


class VolcanoPlotNode(Node):
    """ volcano.R wrap → log2FC vs Combined Score volcano PNG."""
    name = "volcano_plot"

    def run(self) -> None:
        from viz_report_mcp.tools.integrated_plots import register_integrated_plot_tools
        tools = capture_mcp_tools(register_integrated_plot_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "volcano.png"
        res = json.loads(tools["plot_volcano"](
            integrated_csv=self.inputs["integrated_csv"],
            output_png=str(out_png),
            top_n=self.params.get("top_n", 10),
        ))
        if res.get("status") != "ok":
            raise RuntimeError(f"plot_volcano failed: {res.get('stderr','')[:300]}")
        self.outputs = {"volcano_png": str(out_png)}


class HeatmapPlotNode(Node):
    """Heatmap.R wrap → top-N up + top-N down genes' RSEM counts heatmap."""
    name = "heatmap_plot"

    def run(self) -> None:
        from viz_report_mcp.tools.integrated_plots import register_integrated_plot_tools
        tools = capture_mcp_tools(register_integrated_plot_tools)
        out_dir = Path(self.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "heatmap.png"
        res = json.loads(tools["plot_heatmap"](
            integrated_csv=self.inputs["integrated_csv"],
            deseq_csv=self.inputs["deseq_csv"],
            counts_txt=self.inputs["counts_txt"],
            output_png=str(out_png),
            top_n=self.params.get("top_n", 10),
            group_a_label=self.params.get("group_a_label", "OC2"),
            group_b_label=self.params.get("group_b_label", "Cyc"),
            samples_a=self.inputs.get("samples_a"),
            samples_b=self.inputs.get("samples_b"),
            color_a=self.params.get("color_a", "gold"),
            color_b=self.params.get("color_b", "purple"),
        ))
        if res.get("status") != "ok":
            raise RuntimeError(f"plot_heatmap failed: {res.get('stderr','')[:300]}")
        self.outputs = {"heatmap_png": str(out_png)}
