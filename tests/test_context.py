from src.context import AgentResult, ExecutionContext
from src.types import Event, Message


def test_execution_context_tracks_events_and_steps():
    context = ExecutionContext()
    event = Event(
        execution_id=context.execution_id,
        author="user",
        content=[Message(role="user", content="hello")],
    )

    context.add_event(event)
    context.increment_step()

    assert context.events == [event]
    assert context.current_step == 1
    assert AgentResult(output="done", context=context).status == "complete"
