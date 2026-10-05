"""
Tools exposed by external MCP servers (local or remote).

NOTE: `langchain.mcp.MCPAdapter` is carried over as-is from the original
project. Depending on your installed langchain/langgraph version, MCP
integration is more commonly done via the separate `langchain-mcp-adapters`
package (`langchain_mcp_adapters.client.MultiServerMCPClient`). If this import
fails for you, that's almost certainly why - see the README's "Known caveats"
section.
"""
from langchain_core.tools import BaseTool

from app.config import MCP_CONFIG


async def load_mcp_tools() -> list[BaseTool]:
    try:
        from langchain.mcp import MCPAdapter  # imported lazily - see module docstring

        async with MCPAdapter(MCP_CONFIG) as adapter:
            return await adapter.list_tools()
    except Exception as e:
        print(f"[MCP] Failed to load tools: {e}")
        return []
