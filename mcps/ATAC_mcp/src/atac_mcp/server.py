"""ATAC-seq MCP Server — main entry point."""
import json
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "atac-mcp",
    instructions="ATAC-seq pipeline with AI-powered automation for HPC. "
                 "Provides tools for preprocessing, alignment, filtering, "
                 "peak calling, annotation, DAStk analysis, and AI-driven "
                 "sample detection, QC, group classification, and more.",
)

from atac_mcp.tools.pipeline import register_pipeline_tools
from atac_mcp.tools.ai_features import register_ai_tools
from atac_mcp.tools.executor import LocalExecutor

executor = LocalExecutor(
    work_dir=os.environ.get("ATAC_WORKDIR", "/BIO1/MRAG"),
)

register_pipeline_tools(mcp, executor)
register_ai_tools(mcp, executor)
# MCP step 2a (2026-10-12): super-enhancer calling (ROSE), GIGGLE TF overlap, condition-specific peak sets
from atac_mcp.tools.super_enhancers import register_super_enhancer_tools
from atac_mcp.tools.giggle import register_giggle_tools
from atac_mcp.tools.peak_sets import register_peak_set_tools
register_super_enhancer_tools(mcp)
register_giggle_tools(mcp)
register_peak_set_tools(mcp)


@mcp.tool()
def atac_status(project_dir: str = "") -> str:
    """Show pipeline checkpoint: completed steps, pending steps, sample list.

    Args:
        project_dir: Absolute path to the ATAC project directory.
    """
    checkpoint = executor.load_checkpoint(project_dir or executor.work_dir)
    return json.dumps(checkpoint, indent=2, ensure_ascii=False)


def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
