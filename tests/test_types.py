import pytest
from pydantic import ValidationError

from src.types import Event, Message, ToolCall, ToolResult


def test_content_models_and_event_defaults():
    items = [
        Message(role="user", content="hello"),
        ToolCall(tool_call_id="1", name="echo", arguments={"text": "hi"}),
        ToolResult(
            tool_call_id="1", name="echo", status="success", content=["hi"]
        ),
    ]
    event = Event(execution_id="execution", author="agent", content=items)

    assert [item.type for item in event.content] == [
        "message", "tool_call", "tool_result"
    ]
    assert event.id
    assert event.timestamp > 0


def test_message_rejects_an_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="tool", content="invalid")
