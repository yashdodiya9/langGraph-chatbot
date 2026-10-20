"""Tiny local MCP server (stdio) with two arithmetic tools, for trying ENABLE_MCP_TOOLS=true."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arith")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    mcp.run()
