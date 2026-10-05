"""
End-to-end tests of the real compiled graph + real sqlite checkpointer + real
thread metadata, with only the LLM (and the outbound stock-price HTTP call)
replaced by fakes. No network, no API keys.

They cover the behaviours this project is actually about:
  plain chat, clarification interrupts, approve / edit-and-approve / reject,
  read-only tools skipping approval, auto-titles, pinning and deleting chats.
"""
import time
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.backend import graph as graph_mod
from app.backend import human_review as review_mod
from app.backend.async_runtime import run_async
from app.backend.persistence import thread_meta
from app.backend.tools import rag, stock
from tests.fakes import FakeHTTPResponse, ScriptedChatModel, tool_call


@pytest.fixture
def stock_http(monkeypatch):
    """Replace the Alpha Vantage HTTP call; records every URL requested."""
    urls = []

    def fake_get(url, timeout=None):
        urls.append(url)
        return FakeHTTPResponse()

    monkeypatch.setattr(stock.requests, "get", fake_get)
    return urls


def use_llm(monkeypatch, *responses):
    model = ScriptedChatModel(responses=list(responses))
    monkeypatch.setattr(graph_mod, "llm_with_tools", model)
    return model


@pytest.fixture
def tid():
    return f"test-{uuid.uuid4()}"


# --------------------------------------------------------------------------- helpers
def cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def run(input_data, thread_id):
    """Drive one graph turn the same way the Streamlit frontend does."""

    async def go():
        return [c async for c, _ in graph_mod.chatbot.astream(input_data, config=cfg(thread_id), stream_mode="messages")]

    return run_async(go())


def say(text, thread_id):
    return run({"messages": [HumanMessage(content=text)]}, thread_id)


def pending_interrupt(thread_id):
    """Same logic as get_pending_interrupt() in the frontend."""
    state = graph_mod.chatbot.get_state(cfg(thread_id))
    if not state.next:
        return None
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def messages(thread_id):
    return graph_mod.chatbot.get_state(cfg(thread_id)).values.get("messages", [])


# ------------------------------------------------------------------------------ tests
def test_plain_chat_is_persisted(monkeypatch, tid):
    use_llm(monkeypatch, AIMessage(content="Hi there!"))
    say("hello", tid)

    assert pending_interrupt(tid) is None
    msgs = messages(tid)
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    assert msgs[1].content == "Hi there!"
    # the checkpointer really stored it, so the thread shows up in the sidebar listing
    assert tid in {t["thread_id"] for t in thread_meta.retrieve_all_threads()}


def test_read_only_tool_runs_without_approval(monkeypatch, tid, stock_http):
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", set())
    use_llm(monkeypatch, tool_call("get_stock_price", {"symbol": "AAPL"}), AIMessage(content="AAPL is 123.45"))
    say("price of AAPL?", tid)

    assert pending_interrupt(tid) is None  # never paused
    assert len(stock_http) == 1 and "symbol=AAPL" in stock_http[0]
    assert messages(tid)[-1].content == "AAPL is 123.45"


def test_clarification_interrupt_fills_missing_argument(monkeypatch, tid, stock_http):
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", set())
    use_llm(monkeypatch, tool_call("get_stock_price", {"symbol": ""}), AIMessage(content="Done"))

    say("what's the stock price?", tid)
    interrupt = pending_interrupt(tid)
    assert interrupt["kind"] == "clarification" and interrupt["field"] == "symbol"
    assert stock_http == []  # tool must NOT run before the user answers

    run(Command(resume="TSLA"), tid)
    assert pending_interrupt(tid) is None
    assert "symbol=TSLA" in stock_http[0]
    ai_with_call = next(m for m in messages(tid) if isinstance(m, AIMessage) and m.tool_calls)
    assert ai_with_call.tool_calls[0]["args"]["symbol"] == "TSLA"
    assert messages(tid)[-1].content == "Done"


def test_sensitive_tool_approve_with_edited_args(monkeypatch, tid, stock_http):
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", {"get_stock_price"})
    use_llm(monkeypatch, tool_call("get_stock_price", {"symbol": "AAPL"}), AIMessage(content="Done"))

    say("price?", tid)
    interrupt = pending_interrupt(tid)
    assert interrupt["kind"] == "approval" and interrupt["args"] == {"symbol": "AAPL"}
    assert stock_http == []

    run(Command(resume={"action": "approve", "args": {"symbol": "MSFT"}}), tid)  # user edits AAPL -> MSFT
    assert len(stock_http) == 1 and "symbol=MSFT" in stock_http[0]
    assert messages(tid)[-1].content == "Done"


def test_sensitive_tool_reject_skips_tool(monkeypatch, tid, stock_http):
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", {"get_stock_price"})
    model = use_llm(monkeypatch, tool_call("get_stock_price", {"symbol": "AAPL"}), AIMessage(content="Okay, cancelled."))

    say("price?", tid)
    assert pending_interrupt(tid)["kind"] == "approval"

    run(Command(resume={"action": "reject"}), tid)
    assert stock_http == []  # tool never ran
    assert pending_interrupt(tid) is None
    assert messages(tid)[-1].content == "Okay, cancelled."

    # what the LLM was shown on its follow-up call after the rejection must be a VALID
    # assistant(tool_calls) -> tool(result) pair, or OpenAI-style APIs (Groq) reject the request
    follow_up = model.received[-1]
    ai = next(m for m in follow_up if isinstance(m, AIMessage))
    tool_msg = next(m for m in follow_up if isinstance(m, ToolMessage))
    assert [c["id"] for c in ai.tool_calls] == [tool_msg.tool_call_id] == ["call_1"]
    assert "cancelled" in tool_msg.content.lower()


def _two_calls():
    """One model message requesting a read-only tool AND a sensitive one."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "retrieve_pdf_content", "args": {"query": "x"}, "id": "c1"},
            {"name": "get_stock_price", "args": {"symbol": "AAPL"}, "id": "c2"},
        ],
    )


def test_sensitive_tool_is_gated_even_when_not_the_first_call(monkeypatch, tid, stock_http):
    """Regression: only the FIRST tool call used to be reviewed, so a sensitive call
    in position 2+ ran without any approval prompt."""
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", {"get_stock_price"})
    use_llm(monkeypatch, _two_calls(), AIMessage(content="done"))

    say("do both", tid)
    assert pending_interrupt(tid)["kind"] == "approval"
    assert stock_http == []  # not executed before approval

    run(Command(resume={"action": "approve", "args": {"symbol": "AAPL"}}), tid)
    assert len(stock_http) == 1
    assert messages(tid)[-1].content == "done"


def test_partial_rejection_runs_only_approved_calls(monkeypatch, tid, stock_http):
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", {"get_stock_price"})
    model = use_llm(monkeypatch, _two_calls(), AIMessage(content="done"))

    say("do both", tid)
    run(Command(resume={"action": "reject"}), tid)

    assert stock_http == []  # the rejected sensitive call never ran
    assert pending_interrupt(tid) is None
    follow_up = model.received[-1]
    ai = next(m for m in follow_up if isinstance(m, AIMessage))
    tool_ids = [m.tool_call_id for m in follow_up if isinstance(m, ToolMessage)]
    assert [c["id"] for c in ai.tool_calls] == tool_ids == ["c1"]  # read-only call ran, pairs valid


def test_title_generation_pin_and_delete_lifecycle(monkeypatch, tid):
    use_llm(monkeypatch, AIMessage(content="Hi!"))
    monkeypatch.setattr(thread_meta, "model", ScriptedChatModel(responses=[AIMessage(content='"Greeting Chat"')]))

    say("hello", tid)
    thread_meta.register_thread(tid)
    thread_meta.generate_and_save_title(tid, "hello")  # fire-and-forget

    def row():
        return next((t for t in thread_meta.retrieve_all_threads() if t["thread_id"] == tid), None)

    deadline = time.time() + 5
    while time.time() < deadline and row()["title"] == "New Chat":
        time.sleep(0.05)
    assert row()["title"] == "Greeting Chat"  # quotes stripped

    thread_meta.pin_thread(tid, True)
    assert row()["pinned"] is True

    # give it an indexed PDF, then delete the chat: everything must go
    rag.PDF_VECTORSTORES[tid] = object()
    thread_meta.delete_thread(tid)
    assert row() is None
    assert tid not in rag.PDF_VECTORSTORES
    assert graph_mod.chatbot.get_state(cfg(tid)).values == {}


def test_title_falls_back_to_message_when_llm_fails(monkeypatch):
    monkeypatch.setattr(thread_meta, "model", ScriptedChatModel(responses=[AIMessage(content="x")], fail=True))
    title = run_async(thread_meta._agenerate_title("Explain how vector databases work in simple terms please"))
    assert title.startswith("Explain how vector databases work")
    assert len(title) <= 41
