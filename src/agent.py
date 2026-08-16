# --------------------------------------------------------------------------- #
# Basic OperationaL Agent class
# --------------------------------------------------------------------------- #

"""
Basic ReAct agent
Differences from final version:
  - __init__: only output_type (callbacks, session, code_execution, sub_agents
      not present)
  - run(): session not present, confirmation not present, code_env not present
    transfer not present
  - step(): before_llm_callbacks not present
  - act(): callbacks not present, confirmation check not present
  - setup_tools(): only final_answer (execute_python, transfer,
    memory not present)
  - _prepare_llm_request(): sandbox/skills prompt not present
"""

import argparse  # noqa: I001
import asyncio  # noqa: I001, RUF100
import logging  # noqa: I001, RUF100
from collections.abc import Callable
from typing import Any, Sequence, cast  # noqa: UP035

from dotenv import load_dotenv
from pydantic import BaseModel

from src.llm import LlmClient, LlmRequest, LlmResponse
from src.tools import BaseTool, FunctionTool
from src.tools import format_tool_definition, search_web
from src.calculator import calculator
from src.types import Event, Message, ToolCall, ToolResult
from src.context import ExecutionContext, AgentResult

# --------------------------------------------------------------------------- #
# Environment setup and Logging
# --------------------------------------------------------------------------- #

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Agent class
# --------------------------------------------------------------------------- #
class Agent:
    """Tool-calling agent with ReAct loop."""

    def __init__(
        self,
        model: LlmClient,
        tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        output_type: type[BaseModel] | None = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_type = output_type
        self.output_tool_name: str | None = None
        self.tools = self._setup_tools(list(tools or []))

    # Core loop
    async def run(self,
                  user_input: str | None = None,
                  context: ExecutionContext | None = None,
                  verbose: bool = False,
                  ) -> AgentResult:
        """
        Execute the agent.
        """
        if context is None:
            context = ExecutionContext()

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        while (
            not context.final_result
            and context.current_step < self.max_steps
        ):
            await self.step(context, verbose=verbose)

            if context.events:
                last_event = context.events[-1]
                if self._is_final_response(last_event):
                    context.final_result = \
                        self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)

    # Step and action methods
    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> None:
        """Perform one think-act cycle."""
        llm_request = self._prepare_llm_request(context)
        llm_response = await self.think(llm_request)
        if llm_response.error_message:
            raise RuntimeError(f"LLM request failed: {llm_response.error_message}")

        if verbose:
            self._log_response(llm_response)

        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        tool_calls = cast(
            list[ToolCall],
            [c for c in llm_response.content if isinstance(c, ToolCall)],
        )
        if tool_calls:
            await self.act(context, tool_calls)

        context.increment_step()

    # Think and act methods
    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: list[ToolCall],
    ) -> None:
        """Execute the tools requested by the LLM."""
        tools_dict = {t.name: t for t in self.tools}
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

            tool_obj = tools_dict[tool_call.name]

            try:
                output = await tool_obj(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                ))
            except (TypeError, ValueError, KeyError, RuntimeError) as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                ))

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

    # Internal methods
    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Build an LlmRequest from the current context."""
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        instructions = []
        if self.instructions:
            instructions.append(self.instructions)

        if self.output_tool_name:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=instructions,
            # Revalidate at the LLM boundary. This also avoids class identity
            # mismatches after reloading modules in an interactive session.
            contents=[item.model_dump() for item in flat_contents],
            tools=self.tools,
            tool_choice=tool_choice,
        )

    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                ):
                    return True
            return False

        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult)
                               for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> Any:
        """Extract the final result from an event."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content
                ):
                    return item.content[0]

        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _setup_tools(
        self,
        tools: list[BaseTool | Callable[..., Any]],
    ) -> list[BaseTool]:
        """Prepare the tools list, including dynamic tools."""
        prepared_tools = [
            item if isinstance(item, BaseTool) else FunctionTool(item)
            for item in tools
        ]

        if self.output_type is not None:
            output_schema = self.output_type.model_json_schema()
            output_schema.pop("title", None)
            output_schema.pop("$defs", None)

            tool_definition = format_tool_definition(
                "final_answer",
                "Return the final structured answer matching the schema.",
                {
                    "type": "object",
                    "properties": {"output": output_schema},
                    "required": ["output"],
                },
            )

            captured_type = self.output_type

            def _parse_output(output) -> str | BaseModel:
                if isinstance(output, dict):
                    return captured_type.model_validate(output)
                return output

            final_answer_tool = FunctionTool(
                func=_parse_output,
                name="final_answer",
                description=(
                    "Return the final structured answer matching the "
                    "required schema."
                ),
                tool_definition=tool_definition,
            )
            prepared_tools.append(final_answer_tool)
            self.output_tool_name = "final_answer"

        return prepared_tools

    def _log_response(self, response: LlmResponse):
        """Log LLM response for verbose mode."""
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] "
                            f"Tool call: {item.name}({item.arguments})")


# --------------------------------------------------------------------------- #
# Test function for the agent
# --------------------------------------------------------------------------- #

test_query = \
    """
    If Eliud Kipchoge could maintain his official marathon world-record
    time record-making marathon pace indefinitely, how many thousand
    hours would it take him to run the distance between the Earth and
    the Moon at its closest approach?
    Please use the minimum perigee value on the Wikipedia page
    for the Moon when carrying out your calculation. Round your
    result to the nearest thousand hours.
    """


def test_agent():
    """Test the agent with a simple example."""

    # Initialize the agent
    llm_client = LlmClient()
    tools = [calculator, search_web]
    agent = Agent(model=llm_client, tools=tools,
                  instructions="You are an assistant.")

    async def run_examples():
        """Run every request on the same event loop."""
        result1 = await agent.run(user_input="What is 2 + 2?")
        result2 = await agent.run(user_input="What is 1255 * 1255?")
        result3 = await agent.run(user_input=test_query)
        return result1, result2, result3

    result1, result2, result3 = asyncio.run(run_examples())

    # Print the final result
    print("\nFirst Result:", result1.output)
    print("\nSecond Result:", result2.output)
    print("\nThird Result:", result3.output)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the agent from the command line."""
    parser = argparse.ArgumentParser(
        description="Run a query through the tool-calling agent.",
    )
    parser.add_argument("query", help="Query to send to the agent")
    parser.add_argument(
        "--model",
        help=(
            "Model name. Defaults to OLLAMA_DEFAULT_MODEL for Ollama or "
            "OPENAI_DEFAULT_MODEL for OpenAI."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        help="LLM provider. Defaults to LLM_PROVIDER, or Ollama when unset.",
    )
    args = parser.parse_args(argv)

    client = LlmClient(model=args.model, provider=args.provider)
    agent = Agent(
        model=client,
        tools=[calculator, search_web],
        instructions="You are a helpful assistant.",
    )
    result = asyncio.run(agent.run(user_input=args.query))
    if result.output is not None:
        print(result.output)


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------- #
# End of File
# -------------------------------------------------------------------------- #
