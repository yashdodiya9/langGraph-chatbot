"""
Dedicated background asyncio event loop.

Streamlit itself is synchronous, but the LangGraph graph and its checkpointer
are async. Rather than spinning up a new event loop per call (which breaks
long-lived resources like the sqlite connections), we start ONE loop in a
background thread the first time this module is imported, and route every
coroutine through it.
"""
import asyncio
import threading

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    """Schedule a coroutine on the background loop; returns a concurrent.futures.Future."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Run a coroutine on the background loop and block until it completes."""
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the background loop without waiting for it (fire-and-forget)."""
    return _submit_async(coro)
