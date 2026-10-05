"""
Tests for thread_meta.py's bookkeeping (register / pin / rename / touch / ordering).
The DB is the throwaway file set up in conftest.py. The checkpointer's own thread
listing is stubbed here (it's exercised for real in test_graph_flow.py).
"""
import uuid

import pytest

from app.backend.persistence import thread_meta


async def _async_return(value):
    return value


@pytest.fixture
def ids(monkeypatch):
    """Two unique thread ids, with the checkpointer listing stubbed to return just those."""
    a, b = f"a-{uuid.uuid4()}", f"b-{uuid.uuid4()}"
    monkeypatch.setattr(thread_meta, "_alist_thread_ids", lambda: _async_return([a, b]))
    return a, b


def _by_id(threads):
    return {t["thread_id"]: t for t in threads}


def test_register_creates_default_row(ids):
    a, _ = ids
    thread_meta.register_thread(a)
    row = _by_id(thread_meta.retrieve_all_threads())[a]
    assert row["title"] == "New Chat"
    assert row["pinned"] is False


def test_pin_moves_thread_to_top(ids):
    a, b = ids
    thread_meta.register_thread(a)
    thread_meta.register_thread(b)
    thread_meta.pin_thread(b, True)

    threads = thread_meta.retrieve_all_threads()
    assert threads[0]["thread_id"] == b and threads[0]["pinned"] is True

    thread_meta.pin_thread(b, False)
    assert not _by_id(thread_meta.retrieve_all_threads())[b]["pinned"]


def test_rename_thread(ids):
    a, _ = ids
    thread_meta.register_thread(a)
    thread_meta.rename_thread(a, "My Custom Title")
    assert _by_id(thread_meta.retrieve_all_threads())[a]["title"] == "My Custom Title"


def test_register_is_noop_second_time(ids):
    a, _ = ids
    thread_meta.register_thread(a)
    first = _by_id(thread_meta.retrieve_all_threads())[a]["updated_at"]
    thread_meta.register_thread(a)  # e.g. a Streamlit rerun
    assert _by_id(thread_meta.retrieve_all_threads())[a]["updated_at"] == first


def test_touch_bumps_updated_at_and_reorders(ids):
    a, b = ids
    thread_meta.register_thread(a)
    thread_meta.register_thread(b)
    thread_meta.touch_thread(a)  # a is now the most recently used
    assert thread_meta.retrieve_all_threads()[0]["thread_id"] == a
