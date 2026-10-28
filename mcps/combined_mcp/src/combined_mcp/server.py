"""Combined p-value MCP server — Stouffer integration + Top-N TF selection.

This MCP is the bridge between Step 1 (ATAC DAStk + RNA DESeq2) and
Step 2 (DiffBind / HOMER / TRN). It performs two deterministic operations:

    combine_pvalues   ATAC + RNA p-values -> INTEGRATED.csv (Stouffer's Z)
    pick_topn         INTEGRATED.csv      -> top4 / top6 / top8 TF sets

No LLM reasoning happens here — selection is purely statistical so the
downstream Top-N fork is reproducible across re-runs.
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "combined-mcp",
    instructions=(
        "MRAG combined-score MCP. Bridges ATAC/RNA -> Step 2 fork.\n"
        "Tools:\n"
        "  - combine_pvalues : DAStk md_scores + DESeq2 -> INTEGRATED.csv (Stouffer)\n"
        "  - pick_topn       : INTEGRATED.csv -> Top-N (default [4,6,8]) TF sets\n"
        "  - combine_and_pick: combine_pvalues + pick_topn in one call\n"
        "  - integrated_summary: quick stats on an existing INTEGRATED.csv"
    ),
)

from combined_mcp.tools.pipeline import register_pipeline_tools

register_pipeline_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
