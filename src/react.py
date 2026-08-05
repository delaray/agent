
import asyncio
import inspect
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv
from litellm import acompletion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field

from scratch_agents.tools.helpers import (
    format_tool_definition,
    function_to_input_schema,
)

SEMAPHORE = asyncio.Semaphore(3)

load_dotenv()


def _extract_text_content(result) -> str:
    """Extract plain text from an MCP CallToolResult."""
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


# def test_agent():
#     agent = Agent(
#         model=LlmClient(model="gpt-5.4-mini"),
#         tools=[calculator, search_web],
#         instructions="You are a helpful assistant"
#     )

#     result = await agent.run("What is 1234 * 5678?")

#     return result


# ****************************************************************
# Part 1: Implementing the Agent
# ****************************************************************

class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        tool_definition: Dict[str, Any] | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


# -----------------------------------------------------------------------
# FunctionTool
# -----------------------------------------------------------------------

class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        tool_definition: Dict[str, Any] | None = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters

        name = name or func.__name__
        description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()

        super().__init__(
            name=name,
            description=description,
            tool_definition=tool_definition
        )

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


# -----------------------------------------------------------------------
# Load MCP Tool
# -----------------------------------------------------------------------


async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    """Load tools from an MCP server and convert to FunctionTools."""
    tools = []

    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)

    return tools


# -----------------------------------------------------------------------
# Create MCP Tool
# -----------------------------------------------------------------------


def _create_mcp_tool(mcp_tool, connection: dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""

    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)

    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        }
    }

    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition
    )


# -----------------------------------------------------------------------
# LLMRequest & LLMResponse
# -----------------------------------------------------------------------

class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[str] = None


# -----------------------------------------------------------------------

class LlmResponse(BaseModel):
    """Response object from LLM calls."""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)


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
                t.tool_definition for t in request.tools] if request.tools else None

            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice}
                   if request.tool_choice else {}),
                **self.config
            )

            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))

    def _build_messages(self, request: LlmRequest) -> List[dict]:
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


# ****************************************************************
# Part 2: Implementing the Agent
# ****************************************************************

@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel | None
    context: ExecutionContext


class Agent:
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        output_type: Optional[Type[BaseModel]] = None,  # New parameter
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.output_type = output_type
        self.output_tool_name = None  # Will be set if output_type provided
        self.tools = self._setup_tools(tools or [])

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        if self.output_type is not None:
            @tool(
                name="final_answer",
                description="Return the final structured answer matching the required schema."
            )
            def final_answer(output: self.output_type) -> self.output_type:
                return output

            # Create a copy to avoid modifying the original
            tools = list(tools)
            tools.append(final_answer)
            self.output_tool_name = "final_answer"
        
        return tools

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)
        
        # Determine tool choice strategy
        if self.output_tool_name:
            tool_choice = "required"  # Force tool usage for structured output
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    async def run(self, user_input: str,
                  context: ExecutionContext | None = None
                  ) -> AgentResult:
        # Create or reuse context
        if context is None:
            context = ExecutionContext()

        # Add user input as the first event
        user_event = Event(
            execution_id=context.execution_id,
            author="user",
            content=[Message(role="user", content=user_input)]
        )
        context.add_event(user_event)

        # Execute steps until completion or max steps reached
        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context)

        # Check if the last event is a final response
        last_event = context.events[-1]
        if self._is_final_response(last_event):
            context.final_result = self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)

    def _is_final_response(self, event: Event) -> bool:
        if self.output_tool_name:
            # For structured output: check if final_answer tool succeeded
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"):
                    return True
            return False

        # Original logic for free-text responses
        has_tool_calls = any(isinstance(c, ToolCall)
                             for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult)
                               for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event
                              ) -> str | BaseModel | None:
        if self.output_tool_name:
            # Extract structured output from final_answer tool result
            for item in event.content:
                if (isinstance(item, ToolResult) 
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content):
                    return item.content[0]
        
        # Original logic for free-text responses
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    async def step(self, context: ExecutionContext):
        # Prepare what to send to the LLM
        llm_request = self._prepare_llm_request(context)

        # Get LLM's decision
        llm_response = await self.think(llm_request)

        # Record LLM response as an event
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        # Execute tools if the LLM requested any
        tool_calls = [
            c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            tool_results = await self.act(context, tool_calls)
            # Cast to List[ContentItem] for Event
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=tool_results,  # type: ignore
            )
            context.add_event(tool_event)

            context.increment_step()

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        return await self.model.generate(llm_request)

    async def act(self,
                  context: ExecutionContext,
                  tool_calls: List[ToolCall]
                  ) -> List[ToolResult]:
        tools_dict = {tool.name: tool for tool in self.tools}
        results = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue

            tool = tools_dict[tool_call.name]

            try:
                output = await tool(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                ))

        return results


# -------------------------------------------------------------------------
# Sentiment Analysis Tool Example
# -------------------------------------------------------------------------
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: List[str]


async def analyze_sentiment(text: str) -> SentimentAnalysis:
    agent = Agent(
        model=LlmClient(model="gpt-5.4-mini"),
        tools=[],
        instructions="Analyze the sentiment of the provided text.",
        output_type=SentimentAnalysis
    )
    query = text
    result = await agent.run(query)

    # "positive"
    print(f"Sentiment: {result.output.sentiment}")
    # 0.92
    print(f"Confidence: {result.output.confidence}")
    # ["exceeded expectations", "highly recommend"]
    print(f"Key phrases: {result.output.key_phrases}")

    return result.output


# ****************************************************************
# End of File
# ****************************************************************
