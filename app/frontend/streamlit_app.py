import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, ToolMessageChunk

from app.backend import (
    chatbot,
    generate_and_save_title,
    register_thread,
    retrieve_all_threads,
    submit_async_task,
    touch_thread,
)
from app.frontend.components.hitl_panel import render_pending_interrupt
from app.frontend.components.sidebar import (
    inject_sidebar_css,
    render_new_chat_button,
    render_pdf_uploader,
    render_thread_list,
)


# ********************* UTILITY FUNCTIONS **************************
def generate_thread_id():
    return uuid.uuid4()  # will generate a random thread_id


def add_thread(thread_id):
    # Just make sure a thread_meta row exists; the sidebar always re-reads the full,
    # sorted (pinned-first) thread list straight from the DB on every rerun, so
    # there's nothing else to keep in sync here.
    register_thread(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["processed_pdf_key"] = None
    st.session_state["pending_interrupt"] = None
    st.session_state["active_config"] = None


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])


def _to_display_history(messages):
    """Collapse the graph's raw message list (HumanMessage / tool-call AIMessage /
    ToolMessage / final-answer AIMessage / ...) down to just the turns a user should
    actually see: their messages and the assistant's final text answers. Tool-call
    requests and raw ToolMessage results are internal plumbing - a live turn only
    ever appends the final answer text, so replaying history should match that,
    instead of surfacing raw tool output as if it were another assistant reply."""
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and not m.tool_calls and m.content:
            history.append({"role": "assistant", "content": m.content})
        # ToolMessage and tool-call AIMessages (content is empty/irrelevant) are skipped.
    return history


def switch_to_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state["processed_pdf_key"] = None
    st.session_state["pending_interrupt"] = None
    st.session_state["active_config"] = None
    st.session_state["message_history"] = _to_display_history(load_conversation(thread_id))


def get_pending_interrupt(config):
    """After a stream ends, check whether the graph paused on an interrupt()
    call instead of finishing. Returns the interrupt's payload dict, or None
    if the turn actually completed."""
    state = chatbot.get_state(config)
    if not state.next:
        return None
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def make_stream(input_data, config, status_holder):
    """Builds a generator that feeds chatbot.astream(...) into st.write_stream,
    the same way for a brand-new turn (input_data = {"messages": [...]})
    or a resumed one (input_data = Command(resume=...))."""

    def ai_only_stream():
        event_queue: queue.Queue = queue.Queue()

        async def run_stream():
            try:
                async for message_chunk, metadata in chatbot.astream(
                    input_data,
                    config=config,
                    stream_mode="messages",
                ):
                    event_queue.put((message_chunk, metadata))
            except Exception as exc:
                event_queue.put(("error", exc))
            finally:
                event_queue.put(None)

        submit_async_task(run_stream())

        while True:
            item = event_queue.get()
            if item is None:
                break
            message_chunk, metadata = item
            if message_chunk == "error":
                raise metadata

            # Lazily create & update the SAME status container when any tool runs
            if isinstance(message_chunk, (ToolMessage, ToolMessageChunk)):
                tool_name = getattr(message_chunk, "name", "tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                else:
                    status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

            # Stream ONLY assistant tokens
            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content

    return ai_only_stream


def run_turn(input_data, config):
    """Runs one graph turn (new message or resume), streams it into the chat
    bubble, then either finalizes the assistant message or stashes a pending
    interrupt for the next rerun to display."""
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        ai_message = st.write_stream(make_stream(input_data, config, status_holder)())
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    pending = get_pending_interrupt(config)
    if pending:
        st.session_state["pending_interrupt"] = pending
        st.session_state["active_config"] = config
    else:
        st.session_state["pending_interrupt"] = None
        st.session_state["active_config"] = None
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        touch_thread(config["configurable"]["thread_id"])
    st.rerun()


# ********************* SESSION SETUP **************************
if "message_history" not in st.session_state:  # create an empty message_history if not present
    st.session_state["message_history"] = []

st.session_state["chat_threads"] = retrieve_all_threads()  # re-read every rerun: picks up new titles/pins

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "processed_pdf_key" not in st.session_state:
    st.session_state["processed_pdf_key"] = None

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

if "active_config" not in st.session_state:
    st.session_state["active_config"] = None

add_thread(st.session_state["thread_id"])  # adding thread_id to current session when loading page


# ********************* SIDEBAR UI **************************
st.sidebar.title("LangGraph Chatbot")
inject_sidebar_css()
render_new_chat_button(reset_chat)
render_pdf_uploader()
render_thread_list(st.session_state["chat_threads"], switch_to_thread, reset_chat)


# ********************* MAIN UI **************************
for message in st.session_state["message_history"]:  # loading the conversation history
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # markdown, not st.text - renders tables/code/bold properly

# ---- HITL: render whatever the graph is currently paused on ----
pending = st.session_state.get("pending_interrupt")
if pending:
    render_pending_interrupt(pending, st.session_state["active_config"], run_turn)

# Disable new messages while a decision is pending, to avoid mixing up turns
user_input = None if pending else st.chat_input("Type here")

if user_input:
    is_first_message = len(st.session_state["message_history"]) == 0

    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if is_first_message:
        # Fire-and-forget: names the chat like Claude/ChatGPT do, without blocking this turn.
        # The title lands in the DB a moment later and shows up on the next rerun.
        generate_and_save_title(st.session_state["thread_id"], user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    run_turn({"messages": [HumanMessage(content=user_input)]}, CONFIG)