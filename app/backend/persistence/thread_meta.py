"""
Chat "metadata" - title, pinned flag, timestamps - kept in a side table in
the same sqlite file the checkpointer uses. The checkpointer only knows
about raw graph state, not chat-app concepts like a title or a pin, so we
track those ourselves instead of bolting them onto ChatState.
"""
import asyncio
import datetime

import aiosqlite
from langchain_core.messages import HumanMessage, SystemMessage

from app.backend.async_runtime import run_async, submit_async_task
from app.backend.llm import model
from app.backend.persistence.checkpointer import checkpointer
from app.backend.tools.rag import clear_pdf
from app.config import DB_PATH


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_meta_lock = asyncio.Lock()  # serializes check-then-write on the meta table


async def _init_thread_meta() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(database=DB_PATH)
    cur = await conn.execute("PRAGMA journal_mode=WAL;")  # see checkpointer.py: must fetch + close
    await cur.fetchone()
    await cur.close()
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_meta (
            thread_id  TEXT PRIMARY KEY,
            title      TEXT,
            pinned     INTEGER NOT NULL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    await conn.commit()
    return conn


_meta_conn = run_async(_init_thread_meta())


async def _aupsert_thread_meta(thread_id: str, title: str | None = None, pinned: bool | None = None):
    """Insert a thread_meta row if missing, else update only the fields passed in."""
    thread_id = str(thread_id)
    now = _now_iso()
    async with _meta_lock:
        cur = await _meta_conn.execute("SELECT 1 FROM thread_meta WHERE thread_id = ?", (thread_id,))
        exists = await cur.fetchone()
        if not exists:
            await _meta_conn.execute(
                "INSERT INTO thread_meta (thread_id, title, pinned, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, title or "New Chat", int(bool(pinned)), now, now),
            )
        elif title is not None or pinned is not None:
            fields, params = [], []
            if title is not None:
                fields.append("title = ?")
                params.append(title)
            if pinned is not None:
                fields.append("pinned = ?")
                params.append(int(bool(pinned)))
            fields.append("updated_at = ?")
            params.append(now)
            params.append(thread_id)
            await _meta_conn.execute(f"UPDATE thread_meta SET {', '.join(fields)} WHERE thread_id = ?", params)
        # else: row exists and nothing changed - a no-op, so reopening a thread
        # doesn't reshuffle the sidebar order.
        await _meta_conn.commit()


def register_thread(thread_id: str) -> None:
    """Ensure a thread has a row (default title 'New Chat'). Safe to call every rerun."""
    run_async(_aupsert_thread_meta(str(thread_id)))


def pin_thread(thread_id: str, pinned: bool) -> None:
    run_async(_aupsert_thread_meta(str(thread_id), pinned=pinned))


def rename_thread(thread_id: str, title: str) -> None:
    run_async(_aupsert_thread_meta(str(thread_id), title=title))


async def _atouch_thread(thread_id: str):
    now = _now_iso()
    async with _meta_lock:
        await _meta_conn.execute("UPDATE thread_meta SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
        await _meta_conn.commit()


def touch_thread(thread_id: str) -> None:
    """Bump updated_at after real activity (a completed turn), so 'most recent'
    sorting reflects last-used, not just when the thread/title was first created."""
    run_async(_atouch_thread(str(thread_id)))


async def _adelete_thread(thread_id: str):
    thread_id = str(thread_id)
    await checkpointer.adelete_thread(thread_id)  # wipes all checkpoints/writes for this thread
    async with _meta_lock:
        await _meta_conn.execute("DELETE FROM thread_meta WHERE thread_id = ?", (thread_id,))
        await _meta_conn.commit()
    clear_pdf(thread_id)  # also drop any indexed PDF vectorstore for this thread


def delete_thread(thread_id: str) -> None:
    run_async(_adelete_thread(thread_id))


async def _agenerate_title(first_message: str) -> str:
    """Ask the LLM for a short (3-6 word) chat title, falling back to a truncated
    version of the message itself if the call fails for any reason."""
    try:
        reply = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Summarize the user's message as a short chat title of 3 to 6 words. "
                        "Reply with ONLY the title itself - no quotes, no punctuation at the end, "
                        "no preamble."
                    )
                ),
                HumanMessage(content=first_message[:500]),
            ]
        )
        title = (reply.content or "").strip().strip('"').strip("'")
        return title[:60] if title else (first_message[:40] or "New Chat")
    except Exception:
        return first_message[:40] + ("…" if len(first_message) > 40 else "")


def generate_and_save_title(thread_id: str, first_message: str) -> None:
    """Fire-and-forget: generate a title from the thread's first user message and store it.
    Called from the frontend right after the first message of a new chat is sent, so the
    slow LLM call never blocks the chat turn itself."""
    thread_id = str(thread_id)

    async def _run():
        title = await _agenerate_title(first_message)
        await _aupsert_thread_meta(thread_id, title=title)

    submit_async_task(_run())


async def _alist_thread_ids() -> list[str]:
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):  # None = no filter -> every thread
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


async def _alist_threads_with_meta() -> list[dict]:
    thread_ids = await _alist_thread_ids()

    async with _meta_lock:
        cur = await _meta_conn.execute("SELECT thread_id, title, pinned, updated_at FROM thread_meta")
        rows = await cur.fetchall()
    meta_by_id = {r[0]: {"title": r[1], "pinned": bool(r[2]), "updated_at": r[3]} for r in rows}

    threads = []
    for tid in thread_ids:
        tid = str(tid)
        meta = meta_by_id.get(tid)
        if meta is None:
            # Thread predates the metadata table (or was never registered) - backfill a row.
            await _aupsert_thread_meta(tid)
            meta = {"title": "New Chat", "pinned": False, "updated_at": ""}
        threads.append({"thread_id": tid, **meta})
    return threads


def retrieve_all_threads() -> list[dict]:
    """Returns [{thread_id, title, pinned, updated_at}, ...], pinned threads first,
    then the rest ordered by most recently updated."""
    threads = run_async(_alist_threads_with_meta())
    pinned = sorted([t for t in threads if t["pinned"]], key=lambda t: t["updated_at"], reverse=True)
    unpinned = sorted([t for t in threads if not t["pinned"]], key=lambda t: t["updated_at"], reverse=True)
    return pinned + unpinned
