import asyncio

from scratch_agents.context import ExecutionContext
from src.base import FunctionTool, tool


def test_function_tool_executes_sync_and_context_functions():
    plain = FunctionTool(lambda value: value * 2, name="double")

    async def with_context(context, value):
        return f"{context.execution_id}:{value}"

    contextual = FunctionTool(with_context)
    context = ExecutionContext()

    assert asyncio.run(plain(context, value=4)) == 8
    assert asyncio.run(contextual(context, value="ok")).endswith(":ok")
    assert plain.tool_definition["function"]["name"] == "double"


def test_tool_decorator_supports_options():
    @tool(name="sum_values", description="Add values")
    def add(left: int, right: int = 1):
        return left + right

    assert isinstance(add, FunctionTool)
    assert add.name == "sum_values"
    assert add.description == "Add values"
