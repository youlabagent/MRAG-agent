"""AlphaGenome MCP server — in silico TF binding-site validation.

In the MRAG pipeline this MCP serves Step 8: for each Top-N TF, run a
counterfactual prediction at the TF's top binding peak (motif scramble vs
reference) and report ATAC / RNA / CHIP_TF deltas. The downstream report
LLM compares these deltas with observed DiffBind/DESeq directions.

Heavy deps (alphagenome SDK + numpy) live in
``/BIO1/MRAG/Agent/libs/alphagenome_client/.venv``. The library
``alphagenome_client`` is editable-installed there. Run this MCP with that
venv's interpreter; ``alphagenome-mcp`` console script is wired up by the
project's pyproject.
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "alphagenome-mcp",
    instructions=(
        "MRAG AlphaGenome MCP. In silico TF binding-site counterfactual prediction\n"
        "for validating selected TFs (Top-N from mcp_combined).\n\n"
        "Validation tools:\n"
        "  - counterfactual_at_peak: single peak scramble -> ATAC/RNA/CHIP delta\n"
        "  - batch_validate_tfs    : Top-N TF list -> per-TF counterfactual table\n\n"
        "Infrastructure / admin:\n"
        "  - list_tissues   : prostate cell lines + modality coverage matrix\n"
        "  - check_coverage : verify (cell_line, outputs) combo is supported\n"
        "  - cache_stats    : SQLite cache size + row count\n"
        "  - purge_cache    : drop cache rows older than a timestamp"
    ),
)

from alphagenome_mcp.tools.validation import register_validation_tools
from alphagenome_mcp.tools.admin import register_admin_tools

register_validation_tools(mcp)
register_admin_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
