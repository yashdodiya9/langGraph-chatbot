"""
One-off helper to make sure the sqlite tables exist before first run.
In normal usage this happens automatically the moment the app imports
app.backend, but this is handy for CI smoke-tests or a fresh deployment.

Run from anywhere with:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

# Python puts scripts/ (not the repo root) on sys.path when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backend.persistence.checkpointer import checkpointer  # noqa: E402, F401
from app.backend.persistence.thread_meta import _meta_conn  # noqa: E402, F401
from app.config import DB_PATH  # noqa: E402

if __name__ == "__main__":
    print(f"Checkpointer and thread_meta tables are ready in {DB_PATH}")
