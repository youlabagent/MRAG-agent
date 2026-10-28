"""paper-qa MCP server — retrieval (primary) + incremental indexing (admin).

Design notes:
  - The corpus lives at /BIO1/MRAG/Agent/libs/paperqa/ (Qdrant store + parsed JSONL).
  - mrag_paperqa.Retriever / Passage are imported via sys.path injection
    (the library is not pip-installed; the script directory is added at startup).
  - Qdrant local-mode locks the directory while the server is alive, so all
    index mutations (add/remove/rebuild) must go through this MCP. External
    phase5_embed.py scripts can only run while the MCP server is stopped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `mrag_paperqa` importable without installing it.
PAPERQA_ROOT = Path("/BIO1/MRAG/Agent/libs/paperqa")
sys.path.insert(0, str(PAPERQA_ROOT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "paperqa-mcp",
    instructions=(
        "MRAG paper-qa MCP. Dense retrieval over 55k chunks (BGE-M3 / Qdrant)\n"
        "plus tools for incremental ingestion (papers update over time).\n\n"
        "Retrieval (used by orchestrator + report LLM):\n"
        "  - fetch_literature        : single query -> top-K passages\n"
        "  - fetch_literature_for_tfs: batch query for Top-N TF list\n"
        "  - check_papers_indexed    : check doc_ids against the ledger\n\n"
        "Admin / index updates (run when corpus is refreshed):\n"
        "  - index_status   : collection size, ledger summary, model info\n"
        "  - add_papers     : incremental upsert from parsed JSONL (idempotent)\n"
        "  - remove_paper   : delete all chunks for a doc_id\n"
        "  - rebuild_index  : destructive full rebuild (requires confirm=True)"
    ),
)

from paperqa_mcp.tools.retrieval import register_retrieval_tools
from paperqa_mcp.tools.admin import register_admin_tools

register_retrieval_tools(mcp)
register_admin_tools(mcp)
# MCP step 2a (2026-10-12): the curated targets' literature retrieval and the evidence gate's PubMed rescue
from paperqa_mcp.tools.evidence import register_evidence_tools
register_evidence_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
