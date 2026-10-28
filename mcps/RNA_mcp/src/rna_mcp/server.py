"""RNA-seq MCP Server — main entry point."""
import json
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "rna-mcp",
    instructions=(
        "RNA-seq pipeline with AI-powered automation for HPC. "
        "Provides tools for preprocessing (trim_galore+fastqc), STAR alignment, "
        "RSEM quantification, pydeseq2 differential expression analysis, and "
        "MA/Volcano/Heatmap visualisation. AI features include sample detection, "
        "read format/strandedness/genome detection, treatment-control classification, "
        "DE contrast inference, STAR option recommendation, QC auto-judgment, "
        "Ensembl-to-symbol annotation, checkpointing, and resource configuration."
    ),
)

from rna_mcp.tools.pipeline import register_pipeline_tools
from rna_mcp.tools.ai_features import register_ai_tools
from rna_mcp.tools.executor import LocalExecutor

executor = LocalExecutor(
    work_dir=os.environ.get("RNA_WORKDIR", "/BIO1/MRAG"),
)

register_pipeline_tools(mcp, executor)
register_ai_tools(mcp, executor)
# MCP step 2a (2026-10-12): limma group DE for processed RNA matrices
from rna_mcp.tools.matrix_de import register_matrix_de_tools
register_matrix_de_tools(mcp)


@mcp.tool()
def rna_status(project_dir: str = "") -> str:
    """Show pipeline checkpoint: completed steps, pending steps, sample list.

    Args:
        project_dir: Absolute path to the RNA project directory.
    """
    checkpoint = executor.load_checkpoint(project_dir or executor.work_dir)
    return json.dumps(checkpoint, indent=2, ensure_ascii=False)


def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
