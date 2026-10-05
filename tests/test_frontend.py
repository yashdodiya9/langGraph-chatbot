"""
Drives the real Streamlit app (via main.py) headlessly with Streamlit's AppTest harness,
using a fake LLM. Covers the UI wiring: chat, auto-title in the sidebar, pin, delete,
and the human-in-the-loop approval panel.
"""
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from app.backend import graph as graph_mod
from app.backend import human_review as review_mod
from app.backend.persistence import thread_meta
from app.backend.tools import stock
from tests.fakes import FakeHTTPResponse, ScriptedChatModel, tool_call

MAIN = str(Path(__file__).resolve().parents[1] / "main.py")


def _open_app():
    return AppTest.from_file(MAIN, default_timeout=30).run()


def _wait_until(at, predicate, timeout=5.0):
    """Titles are generated in the background, so rerun until the sidebar reflects them."""
    end = time.time() + timeout
    while time.time() < end:
        at.run()
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _bubble_text(chat_message):
    """Both a live-streamed reply and replayed history render as markdown now."""
    return (chat_message.markdown or chat_message.text)[0].value


def _button(at, key):
    return next(b for b in at.sidebar.button if b.key == key)


def test_chat_autotitle_pin_and_delete(monkeypatch):
    title = f"UI Title {uuid.uuid4().hex[:6]}"
    monkeypatch.setattr(graph_mod, "llm_with_tools", ScriptedChatModel(responses=[AIMessage(content="hello from fake")]))
    monkeypatch.setattr(thread_meta, "model", ScriptedChatModel(responses=[AIMessage(content=title)]))

    at = _open_app()
    assert not at.exception

    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert _bubble_text(at.chat_message[1]) == "hello from fake"

    # auto-generated title appears in the sidebar
    assert _wait_until(at, lambda: any(title in b.label for b in at.sidebar.button))
    thread_id = str(at.session_state["thread_id"])

    # pin -> label gets the pin marker
    _button(at, f"pin_{thread_id}").click().run()
    assert any(b.label.startswith("📌") and title in b.label for b in at.sidebar.button)

    # delete the open chat -> it disappears and we land on an empty fresh chat
    _button(at, f"del_{thread_id}").click().run()
    assert not any(title in b.label for b in at.sidebar.button)
    assert len(at.chat_message) == 0
    assert not at.exception


def test_history_replay_renders_markdown_not_raw_text(monkeypatch):
    """Regression: the history-replay loop used to call st.text(), which showed tables
    and code blocks as raw pipe/backtick characters instead of rendering them."""
    table = "| Format | Player |\n|---|---|\n| Test | Kohli |"
    monkeypatch.setattr(graph_mod, "llm_with_tools", ScriptedChatModel(responses=[AIMessage(content=table)]))

    at = _open_app()
    at.chat_input[0].set_value("who is the best").run()
    assert _bubble_text(at.chat_message[1]) == table  # true straight after streaming

    at.run()  # a plain rerun now replays message_history instead of live-streaming
    assert not at.exception
    assert len(at.chat_message[1].text) == 0  # not rendered as plain st.text(...)
    assert at.chat_message[1].markdown[0].value == table  # rendered as st.markdown(...)


def test_switching_back_to_a_tool_using_chat_shows_only_the_final_answer(monkeypatch):
    """Regression: reopening a chat via the sidebar used to replay EVERY raw graph
    message - including the empty tool-call AIMessage and the raw ToolMessage result -
    mislabeled as extra "assistant" bubbles, instead of just the final answer the
    user actually saw live."""
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", set())  # read-only tool, no approval needed
    monkeypatch.setattr(stock.requests, "get", lambda url, timeout=None: FakeHTTPResponse())
    monkeypatch.setattr(thread_meta, "model", ScriptedChatModel(responses=[AIMessage(content="Stock Chat")]))
    monkeypatch.setattr(
        graph_mod,
        "llm_with_tools",
        ScriptedChatModel(responses=[tool_call("get_stock_price", {"symbol": "AAPL"}), AIMessage(content="AAPL is 123.45")]),
    )

    at = _open_app()
    at.chat_input[0].set_value("price of AAPL?").run()
    assert [m.name for m in at.chat_message] == ["user", "assistant"]  # exactly one reply, live

    thread_id = str(at.session_state["thread_id"])
    assert _wait_until(at, lambda: any("Stock Chat" in b.label for b in at.sidebar.button))

    # New Chat, then reopen the original thread from the sidebar - this is what triggers replay
    next(b for b in at.sidebar.button if b.label == "New Chat").click().run()
    _button(at, f"open_{thread_id}").click().run()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]  # still just one clean reply
    assert _bubble_text(at.chat_message[-1]) == "AAPL is 123.45"  # not the raw tool JSON/text


def test_approval_panel_resumes_graph(monkeypatch):
    monkeypatch.setattr(review_mod, "SENSITIVE_TOOLS", {"get_stock_price"})
    monkeypatch.setattr(stock.requests, "get", lambda url, timeout=None: FakeHTTPResponse())
    monkeypatch.setattr(thread_meta, "model", ScriptedChatModel(responses=[AIMessage(content="Stocks")]))
    monkeypatch.setattr(
        graph_mod,
        "llm_with_tools",
        ScriptedChatModel(responses=[tool_call("get_stock_price", {"symbol": "AAPL"}), AIMessage(content="AAPL is 123.45")]),
    )

    at = _open_app()
    at.chat_input[0].set_value("price of AAPL?").run()

    assert "get_stock_price" in at.warning[0].value  # approval prompt is showing
    assert [(t.label, t.value) for t in at.text_input] == [("symbol", "AAPL")]  # editable args
    assert len(at.chat_input) == 0  # typing is disabled while a decision is pending

    next(b for b in at.button if "Approve" in b.label).click().run()
    assert not at.exception
    assert len(at.warning) == 0
    assert _bubble_text(at.chat_message[-1]) == "AAPL is 123.45"