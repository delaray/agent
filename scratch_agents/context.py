"""Execution context and result types for the scratch_agents framework."""

from __future__ import annotations  # noqa: I001

import uuid
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel

from src.types import Event, ToolCall


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = field(default_factory=list)
    current_step: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    final_result: str | BaseModel | None = None
    # CH06 session
    session: Any = None
    session_manager: Any = None
    # CH06 long-term memory
    memory_manager: Any = None
    # CH08 code execution
    code_env: Any = None  # E2B Sandbox
    # CH09 agent transfer
    transfer_to: str | None = None
    transfer_tools: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: Any  # str | BaseModel
    context: ExecutionContext
    status: str = "complete"  # "complete" | "pending_confirmation" | "error"
    pending_tool_calls: list = field(default_factory=list)


class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation (CH06 human-in-the-loop)."""
    tool_call: ToolCall
    confirmation_message: str


class ToolConfirmation(BaseModel):
    """User's response to a pending tool call (CH06 human-in-the-loop)."""
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None

# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
