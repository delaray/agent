import asyncio
import pytest
from types import SimpleNamespace

from src import llm
from src.types import Message, ToolCall, ToolResult


def api_response(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=tool_calls or []
        ))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def test_build_messages_preserves_conversation_without_instructions():
    request = llm.LlmRequest(contents=[
        Message(role="user", content="question"),
        ToolCall(tool_call_id="1", name="lookup", arguments={"q": "x"}),
        ToolResult(
            tool_call_id="1", name="lookup", status="success", content=["answer"]
        ),
    ])

    messages = llm.LlmClient("test")._build_messages(request)
    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool"
    ]


def test_parse_tool_call_without_text_content():
    tool_call = SimpleNamespace(
        id="1", function=SimpleNamespace(name="lookup", arguments='{"q": "x"}')
    )
    result = llm.LlmClient("test")._parse_response(api_response(tool_calls=[tool_call]))

    assert isinstance(result.content[0], ToolCall)
    assert result.content[0].arguments == {"q": "x"}
    assert result.usage_metadata == {"input_tokens": 11, "output_tokens": 7}


def test_generate_calls_litellm_and_converts_expected_errors(monkeypatch):
    async def successful(**kwargs):
        assert kwargs["model"] == "fake-model"
        return api_response(content="done")

    monkeypatch.setattr(llm, "acompletion", successful)
    result = asyncio.run(llm.LlmClient("fake-model", provider="openai").generate(
        llm.LlmRequest(contents=[Message(role="user", content="hello")])
    ))
    assert result.content[0].content == "done"

    async def failing(**kwargs):
        raise ValueError("bad request")

    monkeypatch.setattr(llm, "acompletion", failing)
    result = asyncio.run(llm.LlmClient(
        "fake-model", provider="openai"
    ).generate(llm.LlmRequest()))
    assert result.error_message == "bad request"


def test_ollama_is_default_and_uses_environment(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_NETWORK_HOST", "http://ollama.local:11434/")

    client = llm.LlmClient()

    assert client.provider == "ollama"
    assert client.model == "ollama/qwen3:8b"
    assert client.config["api_base"] == "http://ollama.local:11434"


def test_provider_can_be_switched_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-test")

    client = llm.LlmClient(provider="openai")

    assert client.provider == "openai"
    assert client.model == "gpt-test"


def test_invalid_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm.LlmClient("model", provider="unknown")
