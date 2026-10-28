"""DiffBind MCP server — wraps the R Bioconductor DiffBind pipeline.

In the MRAG pipeline this MCP serves Step 5: per-peak differential binding
(FC + p + FDR) for the selected ATAC samples. Output is consumed by
mcp_homer (filter to selected TF binding peaks) and ultimately by the LLM
report stage (top peak FC values feed AlphaGenome validation).

R / DiffBind subprocess is the right tool: there is no Python equivalent
that matches DiffBind's consensus-peak + count-matrix + DESeq2/edgeR
contrast pipeline. The MCP server itself runs in miniconda Python 3.13
(same env as atac_mcp / rna_mcp / combined_mcp).
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "diffbind-mcp",
    instructions=(
        "MRAG DiffBind MCP. Wraps R DiffBind via subprocess. Tools:\n"
        "  - prepare_diffbind_sheet : build a sample-sheet CSV from a list of samples\n"
        "  - run_diffbind           : run the full DiffBind pipeline (slow, ~10-30 min)\n"
        "  - diffbind_status        : summarise an existing DiffBind output CSV\n"
        "  - diffbind_top_peaks     : extract top-N peaks (sorted by FDR / |Fold|)\n"
        "  - diffbind_check_env     : confirm R + DiffBind are installed"
    ),
)

from diffbind_mcp.tools.pipeline import register_pipeline_tools

register_pipeline_tools(mcp)
# MCP step 2a (2026-10-12): edgeR-voom differential accessibility for processed ATAC matrices
from diffbind_mcp.tools.matrix_da import register_matrix_da_tools
register_matrix_da_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
