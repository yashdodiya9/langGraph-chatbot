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

REJECTED_MESSAGE = (
    "The user REJECTED this action, so it was NOT performed. Do not call this tool again "
    "for it unless the user explicitly asks; tell the user it was not done."
)


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
      - approved calls keep their (possibly edited) args and are run by the tools node
      - rejected calls stay in the AI message but are answered right here with a
        REJECTED_MESSAGE ToolMessage, so the model is told explicitly that the user
        declined them (instead of the call silently vanishing, which made the model
        simply ask for the same action again)
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
                    "call_id": call.get("id"),
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
                    "call_id": call.get("id"),
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

    rejected_ids = {c["id"] for c in rejected}
    edited = {c["id"]: c for c in approved}
    # Keep every call (in order) so each one gets exactly one ToolMessage answer.
    all_calls = [edited.get(c["id"], c) for c in tool_calls]
    rejections = [
        ToolMessage(content=REJECTED_MESSAGE, tool_call_id=c["id"], name=c["name"])
        for c in tool_calls
        if c["id"] in rejected_ids
    ]
    return {"messages": [_rebuild_ai_message(last_message, all_calls), *rejections]}


def pending_tool_calls(messages) -> tuple[AIMessage | None, list[dict]]:
    """The latest AI message and those of its tool calls that have no ToolMessage yet."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            answered = {m.tool_call_id for m in messages[i + 1 :] if isinstance(m, ToolMessage)}
            return msg, [c for c in msg.tool_calls if c["id"] not in answered]
    return None, []


def after_review(state: ChatState):
    """Route after human_review: run the tools if any approved call is still unanswered,
    otherwise (everything was rejected) go straight back to the LLM."""
    _, pending = pending_tool_calls(state["messages"])
    return "tools" if pending else "chat_node"
