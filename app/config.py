"""
Central place for environment variables and tunable constants.
Import from here instead of calling os.getenv() in scattered places.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
# Overridable so tests can point at a throwaway sqlite file instead of the real one.
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "chatbot.db"))

# --- LLM / API keys -----------------------------------------------------------
# ChatGroq reads GROQ_API_KEY from the environment itself - just make sure it's set.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

# --- RAG (PDF) settings ---------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "1000"))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "150"))

# --- Human-in-the-loop -----------------------------------------------------------
# Tool names that mutate data and therefore require explicit human approval before running.
SENSITIVE_TOOLS = {"expense_add_expense"}

# --- MCP servers -------------------------------------------------------------------
# Off by default: connecting at import time to unreachable/misconfigured MCP servers
# is the single biggest source of slow startups and flaky tests. Turn this on once
# your servers are actually configured and reachable.
ENABLE_MCP_TOOLS = os.getenv("ENABLE_MCP_TOOLS", "false").lower() == "true"

MCP_CONFIG = {
    "mcpServers": {
        "arith": {
            "command": "python3",
            # Machine-specific - point this at your own local MCP server.
            "args": [os.getenv("MCP_ARITH_SERVER_PATH", "./mcp_servers/arith_server.py")],
        },
        "expense": {
            "transport": "http",
            "url": os.getenv("MCP_EXPENSE_SERVER_URL", "https://splendid-gold-dingo.fastmcp.app/mcp"),
        },
    }
}
