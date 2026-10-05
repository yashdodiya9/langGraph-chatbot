"""
The human-in-the-loop node that sits between chat_node and tools:
  1) fills in any required tool arguments the model left blank   (interrupt -> "clarification")
  2) pauses sensitive tool calls for approve / edit / reject      (interrupt -> "approval")
Read-only tools (search, stock, pdf, arithmetic, listing expenses) skip step 2
entirely and pass straight through.
"""
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.backend.state import ChatState
from app.backend.tools import tools_by_name
from app.config import SENSITIVE_TOOLS


def _missing_required_args(target_tool, args: dict) -> list[str]:
    """Required fields (per the tool's schema) the model left empty/None."""
    if target_tool is None:
        return []
    try:
        schema = target_tool.args_schema
        if schema is None:
            return []
        json_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
        required = json_schema.get("required", [])
        return [f for f in required if args.get(f) in (None, "")]
    except Exception:
        return []


def _rebuild_ai_message(original: AIMessage, tool_calls: list) -> AIMessage:
    """New AIMessage with a new tool_calls list but the same id, so LangGraph's
    add_messages reducer replaces it in place instead of appending."""
    return AIMessage(
        content=original.content,
        tool_calls=tool_calls,
        id=original.id,
        additional_kwargs=original.additional_kwargs,
        response_metadata=original.response_metadata,
    )


async def human_review(state: ChatState):
    """Reviews EVERY tool call in the model's latest message (a model may request several at once).

    Per call:
      1) missing required args  -> interrupt "clarification" and fill them in
      2) sensitive tool         -> interrupt "approval": approve (optionally edited) or reject

    Outcomes:
      - all approved  : the AI message is updated with the (possibly edited) calls -> run tools
      - some rejected : rejected calls are dropped from the AI message, the rest still run
      - all rejected  : the AI message is kept as-is and every call gets a "cancelled" ToolMessage,
                        so the chat history stays a valid assistant(tool_calls) -> tool(result) sequence
                        that the LLM API accepts; the graph then goes straight back to chat_node.
    """
    last_message = state["messages"][-1]
    tool_calls = [dict(c) for c in (getattr(last_message, "tool_calls", None) or [])]
    if not tool_calls:
        return {}

    approved, rejected = [], []
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = dict(call["args"])
        target_tool = tools_by_name.get(tool_name)

        # Ask the user for anything the model left out
        for field in _missing_required_args(target_tool, tool_args):
            tool_args[field] = interrupt(
                {
                    "kind": "clarification",
                    "tool": tool_name,
                    "field": field,
                    "message": f"I need a value for **{field}** to call `{tool_name}`. What should it be?",
                }
            )

        # Sensitive tools need an explicit decision: approve (optionally with edited args) or reject
        if tool_name in SENSITIVE_TOOLS:
            decision = interrupt(
                {
                    "kind": "approval",
                    "tool": tool_name,
                    "args": tool_args,
                    "message": f"About to call `{tool_name}` with these arguments. Approve or reject?",
                }
            )
            if decision.get("action") == "reject":
                rejected.append(call)
                continue
            tool_args = decision.get("args", tool_args)  # edited args if the user changed any

        call["args"] = tool_args
        approved.append(call)

    if not approved:  # everything was rejected: answer each call so the history stays valid
        cancelled = [
            ToolMessage(content="Action cancelled by the user.", tool_call_id=c["id"], name=c["name"])
            for c in rejected
        ]
        return {"messages": cancelled}

    return {"messages": [_rebuild_ai_message(last_message, approved)]}


def after_review(state: ChatState):
    """Route after human_review: run the tools if calls are still pending, otherwise
    (everything was rejected, so the last message is a ToolMessage) go back to the LLM."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "chat_node"
