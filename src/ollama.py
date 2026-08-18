"""Provider configuration for Ollama and OpenAI LLM calls.

LiteLLM uses an ``ollama_chat/`` model prefix for Ollama's chat API. Keeping
that detail here lets the rest of the agent framework use the same message
and tool formats for both providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

LlmProvider = Literal["ollama", "openai"]
DEFAULT_PROVIDER: LlmProvider = "ollama"


@dataclass(frozen=True)
class LlmConnection:
    """Resolved LiteLLM model name and provider-specific keyword arguments."""

    provider: LlmProvider
    model: str
    config: dict[str, Any]


def normalize_ollama_host(host: str) -> str:
    """Return an absolute Ollama base URL, defaulting to port 11434."""
    candidate = host.strip().rstrip("/")
    if "://" not in candidate:
        candidate = f"http://{candidate.lstrip('/')}"

    parsed = urlsplit(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            "OLLAMA_NETWORK_HOST must be a hostname or an http(s) URL, for "
            "example '192.168.1.32' or 'http://192.168.1.32:11434'"
        )

    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"Invalid OLLAMA_NETWORK_HOST: {error}") from error

    if port is None:
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = f"[{hostname}]"
        parsed = parsed._replace(netloc=f"{hostname}:11434")

    return urlunsplit(parsed).rstrip("/")


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
        # The chat route is required for reliable message and tool-call
        # handling. LiteLLM's ``ollama/`` route uses /api/generate, which can
        # return an empty response when tools are supplied.
        if resolved_model.startswith("ollama/"):
            resolved_model = resolved_model.removeprefix("ollama/")
        if not resolved_model.startswith("ollama_chat/"):
            resolved_model = f"ollama_chat/{resolved_model}"

        host = os.getenv("OLLAMA_NETWORK_HOST")
        if host:
            resolved_config.setdefault("api_base", normalize_ollama_host(host))
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
