import asyncio
from types import SimpleNamespace

from src import llm_communication as communication
from src.types import Message, ToolCall


def test_client_builds_and_parses_messages(monkeypatch):
    request = communication.LlmRequest(
        instructions=["help"],
        contents=[Message(role="user", content="hello")],
    )
    client = communication.LlmClient("model")
    assert client._build_messages(request) == [
        {"role": "system", "content": "help"},
        {"role": "user", "content": "hello"},
    ]

    async def fake_completion(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="1",
                    function=SimpleNamespace(name="lookup", arguments="{}"),
                )],
            ))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )

    monkeypatch.setattr(communication, "acompletion", fake_completion)
    result = asyncio.run(client.generate(request))
    assert isinstance(result.content[0], ToolCall)
