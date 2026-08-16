# *****************************************************************************
# Core types for the Agent framework.
# *****************************************************************************

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Message type
# -----------------------------------------------------------------------------

class Message(BaseModel):
    """A text message in the conversation."""

    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


# -----------------------------------------------------------------------------
# ToolCall type
# -----------------------------------------------------------------------------

class ToolCall(BaseModel):
    """LLM's request to execute a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


# -----------------------------------------------------------------------------
# ToolResult type
# -----------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Result from tool execution."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


# -----------------------------------------------------------------------------
# ContentItem type
# -----------------------------------------------------------------------------

ContentItem = Message | ToolCall | ToolResult


# -----------------------------------------------------------------------------
# Event type
# -----------------------------------------------------------------------------

class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = \
        Field(default_factory=lambda: datetime.now().timestamp())  # noqa: DTZ005
    author: str  # "user" or agent name
    content: list[ContentItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
