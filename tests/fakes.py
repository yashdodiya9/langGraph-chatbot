"""Shared test doubles: a scripted fake LLM and a fake Alpha Vantage HTTP response."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order (repeating the last one) and records
    the messages it was sent on each call."""

    responses: list = Field(default_factory=list)
    received: list = Field(default_factory=list)
    calls: int = 0
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail:
            raise RuntimeError("LLM is down")
        self.received.append(list(messages))
        msg = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy(deep=True))])

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class FakeHTTPResponse:
    def __init__(self, price="123.45"):
        self._price = price

    def raise_for_status(self):
        pass

    def json(self):
        return {"Global Quote": {"05. price": self._price}}


