"""
Tools exposed by external MCP servers (local or remote), loaded with
langchain-mcp-adapters (MultiServerMCPClient).

Each tool is renamed "<server>_<tool>" (e.g. expense_add_expense) so names can't
collide across servers and so SENSITIVE_TOOLS in app/config.py can target them.
A server that is missing or unreachable is skipped with a printed warning; the
chatbot still starts with its built-in tools.
"""
from pathlib import Path

from langchain_core.tools import BaseTool

from app.config import MCP_CONFIG


def _connections() -> dict:
    conns = {}
    for name, cfg in MCP_CONFIG["mcpServers"].items():
        if "url" in cfg:
            conns[name] = {"transport": "streamable_http", "url": cfg["url"]}
        elif cfg.get("args") and Path(cfg["args"][0]).exists():
            conns[name] = {"transport": "stdio", "command": cfg["command"], "args": cfg["args"]}
        else:
            print(f"[MCP] Skipping '{name}': server script not found ({cfg.get('args')})")
    return conns


async def load_mcp_tools() -> list[BaseTool]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print("[MCP] langchain-mcp-adapters is not installed: pip install 'langchain-mcp-adapters<0.2'")
        return []

    conns = _connections()
    client = MultiServerMCPClient(conns)
    loaded: list[BaseTool] = []
    for name in conns:
        try:
            server_tools = await client.get_tools(server_name=name)
        except Exception as e:
            print(f"[MCP] Could not load tools from '{name}': {e}")
            continue
        for tool in server_tools:
            tool.name = f"{name}_{tool.name}"
        loaded.extend(server_tools)
    print(f"[MCP] Loaded {len(loaded)} tool(s): {[t.name for t in loaded]}")
    return loaded
