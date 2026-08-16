"""Base tool abstraction for the scratch_agents framework."""

from __future__ import annotations  # noqa: I001

import os  # noqa: I001, RUF100
from tavily import TavilyClient
from dotenv import load_dotenv

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    get_type_hints,
)

from src.context import ExecutionContext

if TYPE_CHECKING:
    from scratch_agents.llm import LlmRequest

# Load environment variables from .env file
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# Tavily Web Search Tool
# -----------------------------------------------------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY",
                           "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")


# -----------------------------------------------------------------------------

class TavilySearchError(Exception):
    """Raised when Tavily returns an error response."""


# -----------------------------------------------------------------------------

def search_web(query: str) -> dict | None:
    """Search the web for the given query."""
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(query, max_results=2, chunks_per_source=2)
    if response.get("error"):
        raise TavilySearchError(f"Tavily API error: {response['error']}")

    return response.get("results")

# Listing 3.14
# pxsrint(search_web("Kipchoge's marathon world record"))


# ----------------------------------------------------------------------------
# Base Tool Abstraction
# -----------------------------------------------------------------------------
class BaseTool(ABC):
    """Abstract base class for all tools."""

    DEFAULT_CONFIRMATION_TEMPLATE = (
        "The agent wants to execute '{name}' with arguments: {arguments}. "
        "Do you approve?"
    )

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = (
            confirmation_message_template
            if confirmation_message_template
            else self.DEFAULT_CONFIRMATION_TEMPLATE
        )

    @property
    def tool_definition(self) -> dict[str, Any] | None:
        return self._tool_definition

    def get_confirmation_message(self, arguments: dict) -> str:
        return self.confirmation_message_template.format(name=self.name,
                                                         arguments=arguments)

    async def process_llm_request(
        self,
        context: ExecutionContext,
        request: LlmRequest,
    ) -> None:
        """
        Hook for tools to modify the LlmRequest before it is sent to the
        LLM. This can be used to add context, modify arguments, or perform
        any other necessary processing. (Listing 6.37/6.38).
        """
        print(f"Processing LLM request for tool '{self.name}' "
              f"with arguments: {request} anbd context: {context}")

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


# -----------------------------------------------------------------------------
# FunctionTool: Wrap a Python function as a tool
# -----------------------------------------------------------------------------
class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
        sandbox_executable: bool = False,
        requires_confirmation: bool = False,
        confirmation_message_template: str = "",
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters
        self.sandbox_executable = sandbox_executable

        if sandbox_executable and self.needs_context:
            raise ValueError(
                f"Tool '{func.__name__}' cannot be sandbox_executable "
                "because it requires 'context' parameter."
            )

        resolved_name = name or func.__name__
        resolved_desc = description or (func.__doc__ or "").strip()

        # Must set name/description before _generate_definition uses them
        super().__init__(
            name=resolved_name,
            description=resolved_desc,
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template,
        )

        # Generate definition after super().__init__ so self.name is available
        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()

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

    def _generate_definition(self) -> dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)

    def get_source_code(self) -> str:
        """Get the source code of the wrapped function (CH08 sandbox)."""
        if not self.sandbox_executable:
            raise ValueError(f"Tool '{self.name}' is not marked "
                             f"as sandbox_executable")
        source = inspect.getsource(self.func)
        lines = source.split('\n')
        filtered_lines = []
        skip_decorator = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('@tool'):
                skip_decorator = True
                if '(' not in stripped or ')' in stripped:
                    skip_decorator = False
                continue
            if skip_decorator:
                if ')' in stripped:
                    skip_decorator = False
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)


# -----------------------------------------------------------------------------
# Decorator to create a FunctionTool
# -----------------------------------------------------------------------------

def tool(func=None, *, name=None, description=None, sandbox_executable=False,
         requires_confirmation=False, confirmation_message=None):
    """Decorator to create a FunctionTool from a function.

    Can be used with or without arguments:
        @tool
        def my_func(...): ...

        @tool(name="custom_name", description="Custom description")
        def my_func(...): ...
    """
    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            sandbox_executable=sandbox_executable,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message or "",
        )

    if func is not None:
        # Called without arguments: @tool
        return decorator(func)
    # Called with arguments: @tool(name=...)
    return decorator


# -----------------------------------------------------------------------------
# Function to convert a Python function to an input schema
# -----------------------------------------------------------------------------

def function_to_input_schema(func) -> dict:
    """Convert a function's signature to a JSON Schema for tool parameters.

    Inspects type hints and docstring to generate the schema.
    """
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001
        # Fallback for closures where get_type_hints can't resolve annotations
        hints = {
            name: param.annotation
            for name, param in inspect.signature(func).parameters.items()
            if param.annotation is not inspect.Parameter.empty
        }
    sig = inspect.signature(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "context"):
            continue

        prop = {}
        hint = hints.get(name)

        if hint == str:
            prop["type"] = "string"
        elif hint == int:
            prop["type"] = "integer"
        elif hint == float:
            prop["type"] = "number"
        elif hint == bool:
            prop["type"] = "boolean"
        elif hint == list or (
            hint
            and (
                hasattr(hint, "__origin__")
                and hint.__origin__ is list
            )
        ):
            prop["type"] = "array"
            # Try to get item type
            if (
                hint is not None
                and hasattr(hint, "__args__")
                and hint.__args__
            ):
                item_type = hint.__args__[0]
                if item_type == str:
                    prop["items"] = {"type": "string"}
                elif item_type == int:
                    prop["items"] = {"type": "integer"}
                elif hasattr(item_type, "model_json_schema"):
                    prop["items"] = item_type.model_json_schema()
        elif hasattr(hint, "model_json_schema"):
            # Pydantic model
            if hint is not None:
                prop = hint.model_json_schema()
            else:
                prop = {"type": "object"}
        else:
            prop["type"] = "string"

        # Add description from docstring if available
        prop["description"] = f"Parameter: {name}"

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


# -----------------------------------------------------------------------------
# Format tool definition in OpenAI function calling format
# -----------------------------------------------------------------------------

def format_tool_definition(name: str, description: str, parameters: dict
                           ) -> dict:
    """Format a tool definition in the OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }


# -----------------------------------------------------------------------------
# Convert a Python function to an OpenAI-format tool definition
# -----------------------------------------------------------------------------

def function_to_tool_definition(func) -> dict:
    """Convert a Python function to an OpenAI-format tool definition.

    Uses the function name, docstring, and type hints.
    """
    name = func.__name__
    description = inspect.getdoc(func) or f"Function: {name}"
    parameters = function_to_input_schema(func)
    return format_tool_definition(name, description, parameters)


# -----------------------------------------------------------------------------
# Execute a tool call using a tool_box mapping
# -----------------------------------------------------------------------------

def tool_execution(tool_box: dict, tool_call) -> str:
    """Execute a tool call using a tool_box mapping.

    Args:
        tool_box: Dict mapping tool names to callables
        tool_call: Tool call object with function.name and function.arguments
    """
    func_name = tool_call.function.name
    if func_name not in tool_box:
        return f"Error: Unknown tool '{func_name}'"

    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return f"Error executing {func_name}: invalid JSON arguments: {e!s}"

    try:
        result = tool_box[func_name](**args)
    except TypeError as e:
        return f"Error executing {func_name}: invalid arguments: {e!s}"

    return str(result)

# ------------------------------------------------------------------------------
# End of File
# -----------------------------------------------------------------------------
