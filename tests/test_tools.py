import asyncio
from types import SimpleNamespace

import pytest

from src import tools
from src.context import ExecutionContext


def test_function_schema_and_definition_cover_supported_types():
    def sample(name: str, count: int, ratio: float = 1.0,
               enabled: bool = True, tags: list[str] | None = None):
        """Sample function."""

    schema = tools.function_to_input_schema(sample)
    properties = schema["properties"]

    assert properties["name"]["type"] == "string"
    assert properties["count"]["type"] == "integer"
    assert properties["ratio"]["type"] == "number"
    assert properties["enabled"]["type"] == "boolean"
    assert schema["required"] == ["name", "count"]
    assert tools.function_to_tool_definition(sample)["function"]["name"] == "sample"


def test_function_tool_decorator_executes_and_formats_confirmation():
    @tools.tool(requires_confirmation=True)
    def greet(name: str):
        """Greet somebody."""
        return f"hello {name}"

    context = ExecutionContext()
    assert asyncio.run(greet(context, name="Ada")) == "hello Ada"
    assert greet.requires_confirmation is True
    assert "greet" in greet.get_confirmation_message({"name": "Ada"})


def test_tool_execution_handles_success_and_errors():
    call = lambda name, arguments: SimpleNamespace(  # noqa: E731
        function=SimpleNamespace(name=name, arguments=arguments)
    )
    assert tools.tool_execution({"double": lambda value: value * 2},
                                call("double", '{"value": 4}')) == "8"
    assert "Unknown tool" in tools.tool_execution({}, call("missing", "{}"))
    assert "invalid JSON" in tools.tool_execution(
        {"double": lambda value: value * 2}, call("double", "{")
    )


def test_search_web_uses_tavily_and_reports_api_errors(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, query, **kwargs):
            return {"results": [query]} if query != "bad" else {"error": "failed"}

    monkeypatch.setattr(tools, "TavilyClient", FakeClient)
    assert tools.search_web("query") == ["query"]
    with pytest.raises(tools.TavilySearchError, match="failed"):
        tools.search_web("bad")
