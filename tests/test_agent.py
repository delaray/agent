import asyncio

from pydantic import BaseModel

import run_agent as run_agent_module
from src.agent import Agent
from src.llm import LlmResponse
from src.tools import FunctionTool
from src.types import Message, ToolCall


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return next(self.responses)


def test_agent_returns_direct_response_and_wraps_callable_tools():
    client = FakeClient([LlmResponse(
        content=[Message(role="assistant", content="four")],
        usage_metadata={"input_tokens": 3, "output_tokens": 1},
    )])
    agent = Agent(client, tools=[lambda value: value], instructions="help")

    result = asyncio.run(agent.run("What is 2 + 2?"))

    assert result.output == "four"
    assert isinstance(agent.tools[0], FunctionTool)
    assert client.requests[0].tool_choice == "auto"
    assert result.context.events[1].metadata["usage"]["input_tokens"] == 3


def test_agent_executes_tool_then_returns_final_response():
    client = FakeClient([
        LlmResponse(content=[ToolCall(
            tool_call_id="1", name="double", arguments={"value": 4}
        )]),
        LlmResponse(content=[Message(role="assistant", content="The answer is 8")]),
    ])

    def double(value: int):
        return value * 2

    result = asyncio.run(Agent(client, tools=[double]).run("calculate"))

    assert result.output == "The answer is 8"
    assert result.context.events[2].content[0].content == [8]
    assert result.context.current_step == 2


def test_agent_records_missing_tool_and_propagates_llm_errors():
    missing = FakeClient([
        LlmResponse(content=[ToolCall(
            tool_call_id="1", name="missing", arguments={}
        )]),
        LlmResponse(content=[Message(role="assistant", content="recovered")]),
    ])
    result = asyncio.run(Agent(missing).run("run"))
    assert result.context.events[2].content[0].status == "error"

    failed = FakeClient([LlmResponse(error_message="offline")])
    try:
        asyncio.run(Agent(failed).run("run"))
    except RuntimeError as error:
        assert "offline" in str(error)
    else:
        raise AssertionError("Agent did not propagate the LLM error")


def test_structured_output_tool_returns_validated_model():
    class Answer(BaseModel):
        value: int

    client = FakeClient([
        LlmResponse(content=[ToolCall(
            tool_call_id="1",
            name="final_answer",
            arguments={"output": {"value": 42}},
        )])
    ])
    result = asyncio.run(Agent(client, output_type=Answer).run("answer"))
    assert result.output == Answer(value=42)
    assert client.requests[0].tool_choice == "required"


def test_main_accepts_query_model_and_provider(monkeypatch, capsys):
    captured = {}

    class CliClient:
        def __init__(self, model=None, provider=None):
            captured.update(model=model, provider=provider)

        async def generate(self, request):
            return LlmResponse(content=[
                Message(role="assistant", content="command-line answer")
            ])

    monkeypatch.setattr(run_agent_module, "LlmClient", CliClient)

    run_agent_module.main([
        "my query", "--model", "local-model", "--provider", "ollama"
    ])

    assert captured == {"model": "local-model", "provider": "ollama"}
    assert "command-line answer" in capsys.readouterr().out
