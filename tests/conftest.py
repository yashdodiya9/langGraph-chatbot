"""
Test-wide environment. This runs BEFORE any `app.*` module is imported, so:
  - every test uses one throwaway sqlite file (never your real data/chatbot.db)
  - no real API keys are needed (ChatGroq only needs *a* key to be constructed;
    the LLM itself is replaced by a fake in the end-to-end tests)
  - MCP is off, so nothing tries to reach the network at import time
"""
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="chatbot_test_"), "test.db")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("ALPHAVANTAGE_API_KEY", "test-key")
os.environ["ENABLE_MCP_TOOLS"] = "false"
