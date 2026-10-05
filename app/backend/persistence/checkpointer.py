"""
LangGraph checkpointer backed by SQLite. Swap this module out for
AsyncPostgresSaver when moving to a multi-user deployment - nothing else
in the codebase needs to change, since everyone else just imports
`checkpointer` from here.
"""
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.backend.async_runtime import run_async
from app.config import DB_PATH


async def _init_checkpointer() -> AsyncSqliteSaver:
    conn = await aiosqlite.connect(database=DB_PATH)
    # WAL lets thread_meta share the same file safely. NOTE: this PRAGMA returns a result
    # row - it must be fetched and the cursor closed, otherwise the statement stays open,
    # holds a lock, and the second connection fails with "database is locked".
    cur = await conn.execute("PRAGMA journal_mode=WAL;")
    await cur.fetchone()
    await cur.close()
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())
