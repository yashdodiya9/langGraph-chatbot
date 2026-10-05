"""
Unit tests for the pure logic inside human_review.py. The interrupt-driving
async function itself is better covered by an end-to-end graph test; these
cover the two helpers it relies on, in isolation.
"""
from langchain_core.messages import AIMessage

from app.backend.human_review import _missing_required_args, _rebuild_ai_message


class _FakeSchema:
    def __init__(self, required):
        self._required = required

    def model_json_schema(self):
        return {"required": self._required}


class _FakeTool:
    def __init__(self, required):
        self.args_schema = _FakeSchema(required)


def test_missing_required_args_flags_empty_and_missing_fields():
    tool = _FakeTool(required=["amount", "category"])
    args = {"amount": "", "category": "food"}

    assert _missing_required_args(tool, args) == ["amount"]


def test_missing_required_args_returns_empty_when_nothing_missing():
    tool = _FakeTool(required=["amount"])
    args = {"amount": 42}

    assert _missing_required_args(tool, args) == []


def test_missing_required_args_handles_tool_without_schema():
    assert _missing_required_args(None, {}) == []


def test_rebuild_ai_message_keeps_id_and_swaps_tool_calls():
    original = AIMessage(content="", tool_calls=[{"name": "old", "args": {}, "id": "1"}], id="msg-1")
    new_calls = [{"name": "new", "args": {"x": 1}, "id": "2"}]

    rebuilt = _rebuild_ai_message(original, new_calls)

    assert rebuilt.id == "msg-1"
    # LangChain normalises tool calls (adds "type": "tool_call"), so compare the fields we set
    assert [(c["name"], c["args"], c["id"]) for c in rebuilt.tool_calls] == [("new", {"x": 1}, "2")]
