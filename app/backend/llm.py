"""
LLM setup: the base model, plus that same model bound to every available tool.
"""
from langchain_groq import ChatGroq

from app.backend.tools import tools
from app.config import GROQ_MODEL

# Plain model - used directly for lightweight, tool-free calls like chat titling.
model = ChatGroq(model=GROQ_MODEL)

# Same model, but aware of every tool the graph can call.
llm_with_tools = model.bind_tools(tools) if tools else model
