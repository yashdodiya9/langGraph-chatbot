"""
Builds and compiles the LangGraph chatbot: chat_node -> (tools_condition) ->
human_review -> tools -> back to chat_node, with a checkpointer for
persistence and resumable interrupts.
"""
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.backend.human_review import _rebuild_ai_message, after_review, human_review, pending_tool_calls
from app.backend.llm import llm_with_tools
from app.backend.persistence.checkpointer import checkpointer
from app.backend.state import ChatState
from app.backend.tools import tools
from app.backend.tools.rag import PDF_VECTORSTORES


async def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]

    # Guard against empty tool-call results (Groq rejects empty content)
    for m in messages:
        if isinstance(m, ToolMessage) and not m.content:
            m.content = "Tool executed but returned no content."

    # If a PDF is indexed for this thread, explicitly tell the model so it doesn't guess
    thread_id = str(config["configurable"].get("thread_id"))
    if thread_id in PDF_VECTORSTORES:
        pdf_notice = SystemMessage(
            content=(
                "A PDF has been uploaded and indexed for this conversation. "
                "Whenever the user's question could be answered from it, you MUST "
                "call the 'retrieve_pdf_content' tool with a relevant search query "
                "before answering. You can also use the web search, stock price, and "
                "calculator tools when helpful. If no document is available, ask the user "
                "to upload a PDF."
            )
        )
        messages = [pdf_notice, *messages]

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None


async def run_approved_tools(state: ChatState, config: RunnableConfig):
    """Runs only the tool calls human_review left unanswered (the approved ones);
    rejected calls were already answered with a 'rejected' ToolMessage."""
    ai_message, pending = pending_tool_calls(state["messages"])
    if not pending:
        return {}
    result = await tool_node.ainvoke({"messages": [_rebuild_ai_message(ai_message, pending)]}, config)
    return {"messages": result["messages"]}


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")

    if tool_node:
        graph.add_node("human_review", human_review)
        graph.add_node("tools", run_approved_tools)
        graph.add_conditional_edges("chat_node", tools_condition, {"tools": "human_review", END: END})
        graph.add_conditional_edges("human_review", after_review, {"tools": "tools", "chat_node": "chat_node"})
        graph.add_edge("tools", "chat_node")
    else:
        graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)


chatbot = build_graph()
