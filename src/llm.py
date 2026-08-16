# ***********************************************************************
# LLM Communication Layer
# ***********************************************************************

from typing import Any  # noqa: I001
import json
from dotenv import load_dotenv
from litellm import acompletion
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from src.types import ContentItem, Message, ToolCall, ToolResult
from src.tools import BaseTool
from src.ollama import LlmProvider, resolve_llm_connection

load_dotenv(override=True)


# -----------------------------------------------------------------------
# LLMRequest
# -----------------------------------------------------------------------

class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: list[str] = Field(default_factory=list)
    contents: list[ContentItem] = Field(default_factory=list)
    # Agent normalizes callables to BaseTool instances. Skipping validation
    # here avoids false failures when modules are reloaded in a notebook and
    # Pydantic sees old and new copies of the same BaseTool class.
    tools: list[SkipValidation[BaseTool]] = Field(default_factory=list)
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
    """LiteLLM client supporting Ollama (default) and OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        provider: LlmProvider | None = None,
        **config,
    ):
        connection = resolve_llm_connection(model, provider, **config)
        self.provider = connection.provider
        self.model = connection.model
        self.config = connection.config

    async def generate(self, request: LlmRequest
                       ) -> LlmResponse:
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
                    messages[-1].setdefault("tool_calls", []).append(
                        tool_call_dict
                    )
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


# -----------------------------------------------------------------------
# Test function for LlmClient
# -----------------------------------------------------------------------

async def test_llm_client():
    """Test the LlmClient with a simple prompt."""
    # Create client
    client = LlmClient()

    # Build request
    request = LlmRequest(
        instructions=["You are a helpful assistant."],
        contents=[Message(
            role="user",
            content="What is 2 + 2?")],
        )

    # Generate response
    response = await client.generate(request)

    # Response contains the answer
    for item in response.content:
        if isinstance(item, Message):
            print(item.content)


# ***********************************************************************
# End of File
# ***********************************************************************
