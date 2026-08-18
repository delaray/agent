"""Interactive Streamlit workbench for the agent framework."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from time import perf_counter
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

from src.agent import Agent
from src.calculator import calculator
from src.context import AgentResult, ExecutionContext
from src.llm import LlmClient
from src.ollama import normalize_ollama_host
from src.tools import FunctionTool, search_web
from src.types import Event, Message, ToolCall, ToolResult
from src.utils import classify_event, format_elapsed, summarize_event, to_jsonable

load_dotenv(override=True)

st.set_page_config(
    page_title="Agent Workbench",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --accent: #73ead5; --ink: #f7fbfa; --muted: #c7d4d2; }
      .stApp { background: radial-gradient(circle at 80% -20%, #173c43 0%, #0b151a 36%, #081015 75%); color: var(--ink); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b21 0%, #091318 100%); border-right: 1px solid #20343a; color: var(--ink); }
      .workbench-kicker { color: var(--accent); font: 700 .72rem/1.2 monospace; letter-spacing: .16em; text-transform: uppercase; }
      .workbench-title { margin: .25rem 0 0; font-size: clamp(2rem, 4vw, 3.6rem); letter-spacing: -.055em; line-height: 1; }
      .workbench-subtitle { color: var(--muted); max-width: 760px; margin-top: .75rem; }
      .status-pill { display: inline-flex; gap: .45rem; align-items: center; border: 1px solid #416169; border-radius: 999px; padding: .25rem .65rem; color: #e5efed; font-size: .78rem; }
      .status-dot { width: .45rem; height: .45rem; background: var(--accent); border-radius: 50%; box-shadow: 0 0 10px var(--accent); }
      [data-testid="stMetric"] { background: rgba(16, 32, 38, .72); border: 1px solid #223b42; padding: .8rem 1rem; border-radius: 12px; }
      [data-testid="stChatMessage"] { border: 1px solid rgba(72, 108, 113, .32); border-radius: 14px; background: rgba(11, 24, 29, .58); }
      .trace-meta { color: #b9cac7; font: .72rem/1.4 monospace; text-transform: uppercase; letter-spacing: .06em; }
      .tool-chip { display: inline-block; border: 1px solid #41666e; color: #c5f3eb; border-radius: 999px; padding: .1rem .5rem; margin: 0 .25rem .25rem 0; font-size: .72rem; }
      div.stButton > button { border-radius: 10px; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "Calculator": {
        "callable": calculator,
        "description": "Deterministic arithmetic with visible operands and results.",
    },
    "Web search": {
        "callable": search_web,
        "description": "Current web research through Tavily.",
    },
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_ollama_models(host: str) -> tuple[list[str], str | None]:
    """Fetch locally available Ollama models for the model selector."""
    try:
        base_url = normalize_ollama_host(host)
        response = httpx.get(f"{base_url}/api/tags", timeout=2.5)
        response.raise_for_status()
        models = sorted(
            model["name"]
            for model in response.json().get("models", [])
            if model.get("name")
        )
        return models, None
    except (ValueError, httpx.HTTPError, KeyError) as error:
        return [], str(error)


def initialize_state() -> None:
    """Initialize workbench state once per browser session."""
    st.session_state.setdefault("runs", [])
    st.session_state.setdefault("provider", "ollama")


def selected_model(provider: str) -> str:
    """Render provider-aware model controls and return the selected model."""
    if provider == "ollama":
        configured = os.getenv("OLLAMA_DEFAULT_MODEL", "")
        host = os.getenv("OLLAMA_NETWORK_HOST", "")
        models, error = fetch_ollama_models(host) if host else ([], "Host not configured")
        if configured and configured not in models:
            models.insert(0, configured)
        choices = models + ["Custom model…"]
        selection = st.selectbox("Model", choices, key="ollama_model_select")
        if error:
            st.caption(f"Model discovery unavailable: {error}")
        if selection == "Custom model…":
            return st.text_input("Custom model name", value=configured)
        return selection

    configured = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")
    presets = list(dict.fromkeys([configured, "gpt-5-mini", "Custom model…"]))
    selection = st.selectbox("Model", presets, key="openai_model_select")
    if selection == "Custom model…":
        return st.text_input("Custom model name", value=configured, key="openai_custom")
    return selection


def render_sidebar() -> dict[str, Any]:
    """Render runtime configuration and return normalized agent settings."""
    with st.sidebar:
        st.markdown("### Runtime")
        provider = st.segmented_control(
            "Provider",
            options=["ollama", "openai"],
            format_func=str.title,
            key="provider",
        ) or "ollama"
        model = selected_model(provider)

        st.markdown("### Capabilities")
        selected_tools = st.multiselect(
            "Enabled tools",
            list(TOOL_REGISTRY),
            default=list(TOOL_REGISTRY),
        )
        for tool_name in selected_tools:
            st.caption(f"**{tool_name}** · {TOOL_REGISTRY[tool_name]['description']}")

        with st.expander("Model parameters", expanded=False):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.3, 0.05)
            top_p = st.slider("Top P", 0.05, 1.0, 0.95, 0.05)
            max_tokens = st.number_input("Max output tokens", 64, 32768, 2048, 64)
            timeout = st.number_input("Request timeout (seconds)", 5, 600, 120, 5)
            seed_enabled = st.toggle("Deterministic seed")
            seed = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=not seed_enabled)

        with st.expander("Agent behavior", expanded=False):
            max_steps = st.slider("Maximum steps", 1, 30, 10)
            instructions = st.text_area(
                "System instructions",
                value="You are a capable, precise assistant. Use tools when they improve the answer.",
                height=120,
            )

        with st.expander("Advanced parameters", expanded=False):
            extra_text = st.text_area(
                "Additional LiteLLM parameters (JSON)",
                value="{}",
                help="Merged into the model request. Standard controls above take precedence.",
            )

        st.divider()
        left, right = st.columns(2)
        left.metric("Runs", len(st.session_state.runs))
        active_tools = right.metric("Tools", len(selected_tools))
        del active_tools
        if st.button("Clear workspace", width="stretch"):
            st.session_state.runs = []
            st.rerun()

    try:
        extra = json.loads(extra_text)
        if not isinstance(extra, dict):
            raise ValueError("Advanced parameters must be a JSON object")
    except (json.JSONDecodeError, ValueError) as error:
        st.sidebar.error(str(error))
        extra = {}

    config = dict(extra)
    config.update(
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(max_tokens),
        timeout=float(timeout),
    )
    if seed_enabled:
        config["seed"] = int(seed)

    return {
        "provider": provider,
        "model": model,
        "tools": selected_tools,
        "instructions": instructions,
        "max_steps": max_steps,
        "model_config": config,
    }


def event_icon(event: Event) -> str:
    return {
        "user": "◉",
        "assistant": "◆",
        "tool_call": "↗",
        "tool_result": "↙",
        "event": "·",
    }[classify_event(event)]


def render_event(event: Event, index: int, previous_time: float | None) -> None:
    """Render one event with full parameter/result inspection."""
    category = classify_event(event)
    elapsed = event.timestamp - previous_time if previous_time is not None else 0
    title = f"{event_icon(event)} {index:02d} · {category.replace('_', ' ').title()} · {summarize_event(event)}"
    with st.expander(title, expanded=category in ("tool_call", "tool_result")):
        timestamp = datetime.fromtimestamp(event.timestamp).astimezone()
        st.markdown(
            f'<div class="trace-meta">{timestamp:%H:%M:%S.%f} · +{format_elapsed(max(elapsed, 0))} · {event.author}</div>',
            unsafe_allow_html=True,
        )
        for item in event.content:
            if isinstance(item, Message):
                st.markdown(item.content)
            elif isinstance(item, ToolCall):
                st.markdown(f"**Tool:** `{item.name}`  \n**Call ID:** `{item.tool_call_id}`")
                st.markdown("**Arguments**")
                st.json(to_jsonable(item.arguments))
            elif isinstance(item, ToolResult):
                state = "success" if item.status == "success" else "error"
                st.markdown(f"**Tool:** `{item.name}` · **Status:** `{state}`  \n**Call ID:** `{item.tool_call_id}`")
                st.markdown("**Result**")
                st.json(to_jsonable(item.content))
        if event.metadata:
            st.markdown("**Metadata**")
            st.json(to_jsonable(event.metadata))
        with st.popover("Raw event"):
            st.json(event.model_dump(mode="json"))


def render_trace(run: dict[str, Any]) -> None:
    """Render metrics and the event timeline for a stored run."""
    events: list[Event] = run["result"].context.events
    tool_calls = sum(
        isinstance(item, ToolCall) for event in events for item in event.content
    )
    usage = {
        key: sum(
            int(event.metadata.get("usage", {}).get(key, 0) or 0)
            for event in events
        )
        for key in ("input_tokens", "output_tokens")
    }
    cols = st.columns(4)
    cols[0].metric("Duration", format_elapsed(run["elapsed"]))
    cols[1].metric("Agent steps", run["result"].context.current_step)
    cols[2].metric("Tool calls", tool_calls)
    cols[3].metric("Tokens", usage["input_tokens"] + usage["output_tokens"])

    st.caption(
        f"{run['provider'].title()} · {run['model']} · execution {run['result'].context.execution_id}"
    )
    previous = None
    for index, event in enumerate(events, start=1):
        render_event(event, index, previous)
        previous = event.timestamp

    st.download_button(
        "Download trace JSON",
        data=json.dumps(to_jsonable(run["result"].context.events), indent=2),
        file_name=f"agent-trace-{run['result'].context.execution_id}.json",
        mime="application/json",
    )


def execute_task(prompt: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Execute one task while publishing events into a Streamlit status box."""
    selected_tools = [
        TOOL_REGISTRY[name]["callable"] for name in settings["tools"]
    ]
    client = LlmClient(
        model=settings["model"],
        provider=settings["provider"],
        **settings["model_config"],
    )
    agent = Agent(
        model=client,
        tools=selected_tools,
        instructions=settings["instructions"],
        max_steps=settings["max_steps"],
    )

    started = perf_counter()
    with st.status("Agent is working…", expanded=True) as status:
        def publish(event: Event) -> None:
            status.write(f"{event_icon(event)} {summarize_event(event)}")

        context = ExecutionContext(event_handlers=[publish])
        try:
            result = asyncio.run(agent.run(prompt, context=context))
        except Exception as error:  # noqa: BLE001
            status.update(label="Run failed", state="error", expanded=True)
            raise error
        status.update(label="Run complete", state="complete", expanded=False)

    return {
        "query": prompt,
        "result": result,
        "elapsed": perf_counter() - started,
        "provider": settings["provider"],
        "model": client.model,
        "tools": settings["tools"],
        "model_config": settings["model_config"],
        "created_at": datetime.now().astimezone().isoformat(),
    }


def main() -> None:
    initialize_state()
    settings = render_sidebar()

    st.markdown('<div class="workbench-kicker">Local agent operations</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="workbench-title">Agent Workbench</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="workbench-subtitle">Configure a model, delegate work, and inspect every decision and tool exchange in one operational view.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-pill"><span class="status-dot"></span>{settings["provider"].title()} · {settings["model"] or "model required"}</span>',
        unsafe_allow_html=True,
    )

    conversation_tab, trace_tab, tools_tab = st.tabs(
        ["Conversation", "Execution trace", "Tool catalog"]
    )

    with conversation_tab:
        if not st.session_state.runs:
            st.info("Start with a task below. Each run will be preserved in this browser session.")
        for run in st.session_state.runs:
            with st.chat_message("user"):
                st.markdown(run["query"])
            with st.chat_message("assistant"):
                st.markdown(str(run["result"].output or "No final response was returned."))
                st.caption(
                    f"{run['provider'].title()} · {run['model']} · {format_elapsed(run['elapsed'])}"
                )

    with trace_tab:
        if st.session_state.runs:
            options = list(range(len(st.session_state.runs) - 1, -1, -1))
            selected = st.selectbox(
                "Run",
                options,
                format_func=lambda index: (
                    f"Run {index + 1} · {st.session_state.runs[index]['query'][:72]}"
                ),
            )
            render_trace(st.session_state.runs[selected])
        else:
            st.info("Execution events will appear here after the first run.")

    with tools_tab:
        st.markdown("#### Available capabilities")
        for name, entry in TOOL_REGISTRY.items():
            tool = FunctionTool(entry["callable"])
            with st.expander(name):
                st.write(entry["description"])
                st.json(tool.tool_definition)

    prompt = st.chat_input("Describe a task for the agent…")
    if prompt:
        if not settings["model"].strip():
            st.error("Select or enter a model before running the task.")
            st.stop()
        try:
            run = execute_task(prompt, settings)
        except Exception as error:  # noqa: BLE001
            st.error(f"Agent run failed: {error}")
        else:
            st.session_state.runs.append(run)
            st.rerun()


if __name__ == "__main__":
    main()
