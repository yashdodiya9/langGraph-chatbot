"""
Aggregates every tool the graph can call: built-in tools plus whatever the
configured MCP servers expose (only attempted if ENABLE_MCP_TOOLS=true).
"""
from app.backend.async_runtime import run_async
from app.backend.tools.mcp_tools import load_mcp_tools
from app.backend.tools.rag import retrieve_pdf_content
from app.backend.tools.search import search_tool
from app.backend.tools.stock import get_stock_price
from app.config import ENABLE_MCP_TOOLS

mcp_tools = run_async(load_mcp_tools()) if ENABLE_MCP_TOOLS else []

tools = [search_tool, get_stock_price, retrieve_pdf_content, *mcp_tools]
tools_by_name = {t.name: t for t in tools}
