# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.8.0 AS uv

FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Resolve the locked production environment in its own cacheable layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=true \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN groupadd --system agent \
    && useradd --system --gid agent --home-dir /app agent

COPY --from=builder --chown=agent:agent /app/.venv /app/.venv
COPY --chown=agent:agent src ./src
COPY --chown=agent:agent scratch_agents ./scratch_agents
COPY --chown=agent:agent .streamlit ./.streamlit
COPY --chown=agent:agent agent_workbench.py run_agent.py ./

USER agent

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "agent_workbench.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
