# ***********************************************************************
# LLM Communication Module
# ***********************************************************************

import json  # noqa: I001
from typing import Any

from litellm import acompletion
from pydantic import BaseModel, ConfigDict, Field

from src.tools import BaseTool
from src.types import ContentItem, Message, ToolCall, ToolResult


# -----------------------------------------------------------------------
# LLMRequest\
# -----------------------------------------------------------------------

class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: list[str] = Field(default_factory=list)
    contents: list[ContentItem] = Field(default_factory=list)
    tools: list[BaseTool] = Field(default_factory=list)
    tool_choice: str | None = None


# -----------------------------------------------------------------------
# LLMResponse
# -----------------------------------------------------------------------

class LlmResponse(BaseModel):
    """Response object from LLM calls."""
    content: list[ContentItem] = Field(default_factory=list)
    error_message: str | None = None
    usage_metadata: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------
# LLM Client
# -----------------------------------------------------------------------

class LlmClient:
    """Client for LLM API calls using LiteLLM."""

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = self._build_messages(request)
            tools = [
                t.tool_definition for t in request.tools
            ] if request.tools else None

            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice}
                   if request.tool_choice else {}),
                **self.config
            )

            return self._parse_response(response)
        except (ValueError, KeyError, RuntimeError) as e:
            return LlmResponse(error_message=str(e))

    def _build_messages(self, request: LlmRequest) -> list[dict]:
        """Convert LlmRequest to API message format."""
        messages = []

        for instruction in request.instructions:
            messages.append({"role": "system", "content": instruction})

            for item in request.contents:
                if isinstance(item, Message):
                    messages.append(
                        {"role": item.role, "content": item.content})

                elif isinstance(item, ToolCall):
                    tool_call_dict = {
                        "id": item.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": json.dumps(item.arguments)
                        }
                    }
                    # Append to previous assistant message if exists
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1].setdefault("tool_calls", []
                                                ).append(tool_call_dict)
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call_dict]
                        })

                elif isinstance(item, ToolResult):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.tool_call_id,
                        "content": str(item.content[0]) if item.content else ""
                    })

        return messages

    def _parse_response(self, response) -> LlmResponse:
        """Convert API response to LlmResponse."""
        choice = response.choices[0]
        content_items = []

        if choice.message.content:
            content_items.append(Message(
                role="assistant",
                content=choice.message.content
            ))

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    content_items.append(ToolCall(
                        tool_call_id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))

        return LlmResponse(
            content=content_items,
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        )

# ***********************************************************************
# End of File
# ***********************************************************************