"""TRN MCP server -- integrate HOMER + DiffBind + DESeq2.

The single output file (``{sample}.TFs.Pval3.txt``) mirrors the reference
analysis's ``target_TFS.py`` STEP4+STEP5 chain: each row carries one
(peak, TF, ATAC FC, RNA FC) tuple of evidence. That single table is the
target-gene extraction artifact, consumed downstream by mcp_viz_report
(Cytoscape, deepTools heatmap, report LLM).
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "trn-mcp",
    instructions=(
        "MRAG TRN MCP. Integrate HOMER filter output + DiffBind + DESeq2\n"
        "into a single evidence table per sample (reference-analysis TFs.Pval3.txt format).\n"
        "Tools:\n"
        "  - build_tfs_pval3 : main integration, produces TFs.Pval3.txt\n"
        "  - tfs_pval3_summary : quick stats on an existing TFs.Pval3.txt"
    ),
)

from trn_mcp.tools.integrate import register_integrate_tools

register_integrate_tools(mcp)
# MCP step 2a (2026-10-12): the processed-matrix Pval2 join; the curated target-prioritisation stages
from trn_mcp.tools.matrix_join import register_matrix_join_tools
from trn_mcp.tools.targets import register_target_tools
register_matrix_join_tools(mcp)
register_target_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
