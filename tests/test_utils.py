from src import utils
from src.types import Event, ToolCall, ToolResult


def test_timing_returns_value_and_reports_elapsed_time(monkeypatch, capsys):
    ticks = iter([60.0, 150.0])
    monkeypatch.setattr(utils, "time", lambda: next(ticks))

    @utils.timing
    def add(left, right=0):
        return left + right

    assert add(2, right=3) == 5
    assert add.__name__ == "add"
    assert capsys.readouterr().out == "add took 1.5 minutes.\n"


def test_trace_utilities_expose_tool_parameters_and_results():
    call = Event(
        execution_id="run",
        author="agent",
        content=[ToolCall(
            tool_call_id="1", name="calculator", arguments={"value": 2}
        )],
    )
    result = Event(
        execution_id="run",
        author="agent",
        content=[ToolResult(
            tool_call_id="1", name="calculator", status="success", content=[4]
        )],
    )

    assert utils.classify_event(call) == "tool_call"
    assert utils.summarize_event(call) == "Requested calculator"
    assert utils.classify_event(result) == "tool_result"
    assert utils.to_jsonable(result)["content"][0]["content"] == [4]
    assert utils.format_elapsed(0.25) == "250 ms"
