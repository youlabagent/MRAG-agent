"""HOMER MCP server — motif discovery + annotation + target TF filter.

In the MRAG pipeline this MCP serves Step 6 and Step 7:
  - Step 6 (sample-level, ~20 min/sample):
      find_motifs   findMotifsGenome.pl  -> peakAnalysis/{sample}/
      annotate_peaks annotatePeaks.pl     -> {sample}.motif.tsv

  - Step 7 (Top-N fork point, cheap per N):
      filter_target_tfs   Keep only columns matching the selected TF gene
                          symbols + promoter rows. Run once per (sample, N).

Heavy operations call HOMER perl binaries via subprocess. The filter and
combine-replicates tools are pure pandas and ship in this MCP itself.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "homer-mcp",
    instructions=(
        "MRAG HOMER MCP. Tools:\n"
        "  Heavy (sample-level, once per sample):\n"
        "    - find_motifs     : findMotifsGenome.pl -> peakAnalysis/{sample}/\n"
        "    - annotate_peaks  : annotatePeaks.pl    -> {sample}.motif.tsv\n"
        "    - concat_known_motifs : merge knownResults/*.motif -> {sample}.motif\n"
        "    - run_homer_pipeline  : find + concat + annotate composite\n"
        "  Light (Top-N fork point):\n"
        "    - filter_target_tfs   : .motif.tsv + TF list -> filtered TSV\n"
        "    - combine_replicates_minmax : 2 replicates -> min-max scaled merged table\n"
        "  Admin:\n"
        "    - homer_check_env : verify HOMER bin + genome installed"
    ),
)

from homer_mcp.tools.pipeline import register_pipeline_tools
from homer_mcp.tools.filter   import register_filter_tools
from homer_mcp.tools.motif_subset import register_motif_subset_tools

register_pipeline_tools(mcp)
register_filter_tools(mcp)
register_motif_subset_tools(mcp)
# MCP step 2a (2026-10-12): SE -> TF motif presence + SE attributes
from homer_mcp.tools.se_motif import register_se_motif_tools
register_se_motif_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
