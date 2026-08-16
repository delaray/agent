"""Provider configuration for Ollama and OpenAI LLM calls.

LiteLLM uses an ``ollama/`` model prefix for Ollama's native API.  Keeping
that detail here lets the rest of the agent framework continue to use the
OpenAI chat-completion message and tool formats for both providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

LlmProvider = Literal["ollama", "openai"]
DEFAULT_PROVIDER: LlmProvider = "ollama"


@dataclass(frozen=True)
class LlmConnection:
    """Resolved LiteLLM model name and provider-specific keyword arguments."""

    provider: LlmProvider
    model: str
    config: dict[str, Any]


def resolve_llm_connection(
    model: str | None = None,
    provider: str | None = None,
    **config: Any,
) -> LlmConnection:
    """Resolve environment and caller settings into a LiteLLM connection.

    ``provider`` takes precedence over ``LLM_PROVIDER``. Ollama is used when
    neither is set. Caller-supplied config always takes precedence over
    environment-derived values.
    """
    selected = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    if selected not in ("ollama", "openai"):
        raise ValueError(
            f"Unsupported LLM provider {selected!r}; expected 'ollama' or 'openai'"
        )

    resolved_config = dict(config)
    if selected == "ollama":
        resolved_model = model or os.getenv("OLLAMA_DEFAULT_MODEL")
        if not resolved_model:
            raise ValueError(
                "An Ollama model is required; pass model=... or set "
                "OLLAMA_DEFAULT_MODEL"
            )
        if not resolved_model.startswith("ollama/"):
            resolved_model = f"ollama/{resolved_model}"

        host = os.getenv("OLLAMA_NETWORK_HOST")
        if host:
            resolved_config.setdefault("api_base", host.rstrip("/"))
    else:
        resolved_model = model or os.getenv("OPENAI_DEFAULT_MODEL")
        if not resolved_model:
            raise ValueError(
                "An OpenAI model is required; pass model=... or set "
                "OPENAI_DEFAULT_MODEL"
            )

    return LlmConnection(
        provider=selected,  # type: ignore[arg-type]
        model=resolved_model,
        config=resolved_config,
    )
