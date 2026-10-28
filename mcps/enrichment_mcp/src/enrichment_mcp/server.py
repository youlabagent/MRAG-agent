"""Enrichment MCP server."""
from fastmcp import FastMCP
from .tools.enrich import register_enrichment_tools

mcp = FastMCP("enrichment-mcp")
register_enrichment_tools(mcp)

if __name__ == "__main__":
    mcp.run()
