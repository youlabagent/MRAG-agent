"""MRAG viz_report MCP server -- interactive HTML report assembly.

Tools (data->JSON-or-PNG):
  - plot_volcano        : Plotly volcano (ATAC/RNA) -> JSON
  - plot_tf_rank        : TF combined-score rank dot plot -> JSON
  - plot_signal_heatmap : deepTools signal heatmap + profile -> PNG (legacy)
  - plot_per_tf_profile : per-TF average ATAC profile 2x3 grid -> JSON
  - build_trn_elements  : cytoscape.js nodes+edges JSON -> JSON
  - plot_locus_track    : pyGenomeTracks locus PNG -> PNG (base64-ready)
  - generate_narrative  : paperqa + report-LLM (Qwen3.6 27B) narrative -> markdown
  - render_report       : assemble all artifacts into interactive HTML
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "viz-report-mcp",
    instructions=(
        "MRAG viz/report MCP. Assemble interactive HTML report (Plotly + cytoscape.js\n"
        "+ DataTables) from upstream pipeline outputs (INTEGRATED.csv, DiffBind_out.csv,\n"
        "DESeq2.gene.csv, TFs.Pval3.txt).\n\n"
        "Tools:\n"
        "  - plot_volcano        : ATAC/RNA volcano (Plotly JSON)\n"
        "  - plot_tf_rank        : TF combined-score rank plot (Plotly JSON)\n"
        "  - plot_signal_heatmap : deepTools signal heatmap+profile (PNG)\n"
        "  - build_trn_elements  : Cytoscape.js graph data\n"
        "  - plot_locus_track    : Locus PNG via pyGenomeTracks helper\n"
        "  - generate_narrative  : paperqa + GLM RAG paper-grounded narrative\n"
        "  - render_report       : assemble interactive HTML report"
    ),
)

from viz_report_mcp.tools.volcano import register_volcano_tools
from viz_report_mcp.tools.tf_rank import register_tf_rank_tools
from viz_report_mcp.tools.signal_heatmap import register_signal_heatmap_tools
from viz_report_mcp.tools.per_tf_profile import register_per_tf_profile_tools
from viz_report_mcp.tools.trn_graph import register_trn_tools
from viz_report_mcp.tools.tracks import register_track_tools
from viz_report_mcp.tools.narrative import register_narrative_tools
from viz_report_mcp.tools.render import register_render_tools
from viz_report_mcp.tools.integrated_scatter import register_integrated_scatter_tools
from viz_report_mcp.tools.orchestrator import register_orchestrator_tools

register_volcano_tools(mcp)
register_tf_rank_tools(mcp)
register_signal_heatmap_tools(mcp)
register_per_tf_profile_tools(mcp)
register_trn_tools(mcp)
register_track_tools(mcp)
register_narrative_tools(mcp)
register_render_tools(mcp)
register_integrated_scatter_tools(mcp)
register_orchestrator_tools(mcp)
# the Section-6 peak heatmaps, the Section-10 SE -> TF -> Target TRN inputs
from viz_report_mcp.tools.peak_heatmap import register_peak_heatmap_tools
from viz_report_mcp.tools.se_trn import register_se_trn_tools
register_peak_heatmap_tools(mcp)
register_se_trn_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
