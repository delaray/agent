from types import SimpleNamespace

from src import mcp


def test_tavily_server_parameters_use_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    params = mcp.tavily_server_params()
    assert params.command == "npx"
    assert params.args == ["-y", "tavily-mcp@latest"]
    assert params.env["TAVILY_API_KEY"] == "secret"


def test_mcp_tools_convert_to_openai_format():
    result = SimpleNamespace(tools=[SimpleNamespace(
        name="lookup", description="Look up data", inputSchema={"type": "object"}
    )])
    converted = mcp.mcp_tools_to_openai_format(result)
    assert converted == [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up data",
            "parameters": {"type": "object"},
        },
    }]
