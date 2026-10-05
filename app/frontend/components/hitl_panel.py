"""
Renders whatever the graph is currently paused on (a clarification question
or a sensitive-tool approval request) and resumes the graph with the user's
decision.
"""
import streamlit as st
from langgraph.types import Command


def render_pending_interrupt(pending, config, run_turn):
    with st.chat_message("assistant"):
        if pending.get("kind") == "clarification":
            # Idea 3: the tool call is missing a required argument - ask for it.
            st.info(pending.get("message", "I need more information."))
            answer = st.text_input("Your answer", key=f"clarify_{pending.get('field')}")
            if st.button("Submit", key="clarify_submit") and answer:
                run_turn(Command(resume=answer), config)

        elif pending.get("kind") == "approval":
            # Ideas 1 & 2: sensitive tool call - approve (optionally edited) or reject.
            st.warning(pending.get("message", "Approval needed."))
            tool_name = pending.get("tool")
            args = pending.get("args", {})
            st.write(f"**Tool:** `{tool_name}`")

            edited_args = {}
            for k, v in args.items():
                raw = st.text_input(k, value=str(v), key=f"edit_{k}")
                try:
                    edited_args[k] = type(v)(raw) if type(v) in (int, float) else raw
                except ValueError:
                    edited_args[k] = raw

            col1, col2 = st.columns(2)
            if col1.button("✅ Approve"):
                run_turn(Command(resume={"action": "approve", "args": edited_args}), config)
            if col2.button("❌ Reject"):
                run_turn(Command(resume={"action": "reject"}), config)
