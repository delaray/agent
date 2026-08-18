import asyncio
from types import SimpleNamespace

from src import react
from src.context import ExecutionContext
from src.types import Message, ToolCall


def test_extract_text_content_ignores_non_text_items():
    result = SimpleNamespace(content=[
        SimpleNamespace(text="first"),
        SimpleNamespace(data=b"ignored"),
        SimpleNamespace(text="second"),
    ])
    assert react._extract_text_content(result) == "first\nsecond"


def test_react_function_tool_executes_sync_function():
    definition = {
        "type": "function",
        "function": {"name": "double", "description": "", "parameters": {}},
    }
    wrapped = react.FunctionTool(
        lambda value: value * 2, name="double", tool_definition=definition
    )
    assert asyncio.run(wrapped(ExecutionContext(), value=5)) == 10


def test_react_tool_generates_its_definition():
    @react.tool
    def greet(name: str):
        """Greet a person."""
        return f"hello {name}"

    assert greet.tool_definition["function"]["name"] == "greet"
    assert greet.tool_definition["function"]["parameters"]["required"] == ["name"]


def test_create_mcp_tool_uses_supplied_schema():
    remote = SimpleNamespace(
        name="lookup", description="Lookup", inputSchema={"type": "object"}
    )
    wrapped = react._create_mcp_tool(remote, {"command": "server"})
    assert wrapped.name == "lookup"
    assert wrapped.tool_definition["function"]["parameters"] == {"type": "object"}


def test_react_agent_returns_direct_response():
    class Client:
        async def generate(self, request):
            return react.LlmResponse(content=[
                Message(role="assistant", content="done")
            ])

    result = asyncio.run(react.Agent(Client(), max_steps=1).run("hello"))
    assert result.output == "done"


def test_react_client_parses_tool_call_without_text():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="1",
                function=SimpleNamespace(name="lookup", arguments='{"q": "x"}'),
            )],
        ))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    parsed = react.LlmClient("model")._parse_response(response)
    assert isinstance(parsed.content[0], ToolCall)
