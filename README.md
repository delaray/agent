# Build an AI Agent from Scratch

Companion code repository for Manning Publications' [*Build an AI Agent from Scratch*](https://www.manning.com/books/build-an-ai-agent-from-scratch).

## Structure

```
scratch_agents/          # Final package (complete through CH10)
  types.py              # Message, ToolCall, ToolResult, Event, ContentItem
  context.py            # ExecutionContext, AgentResult, PendingToolCall, ToolConfirmation
  llm.py                # LlmRequest, LlmResponse, LlmClient
  agent.py              # Agent (ReAct loop)
  rag.py                # Embeddings, chunking, vector search
  callbacks.py          # approval_callback, search_compressor
  planning.py           # Task, create_tasks, reflection
  skills.py             # SkillInfo, discover_skills, generate_skills_prompt
  transfer.py           # create_transfer_tool
  remote.py             # RemoteAgent (A2A)
  a2a_server.py         # MathAgentExecutor
  tools/                # Tool modules
  memory/               # Session, long-term memory, context optimization
  workflows/            # Sequential, Parallel, Loop
  eval/                 # GAIA benchmark, evaluation prompts

notebooks/              # Chapter notebooks
  ch02/                 # LLM API Basics
  ch03/                 # Tools and Function Calling
  ch04/                 # ReAct Agent (+ chapter snapshot code)
  ch05/                 # RAG and File Tools (+ chapter snapshot code)
  ch06/                 # Memory Systems (+ chapter snapshot code)
  ch07/                 # Planning and Reflection
  ch08/                 # Code Execution (+ chapter snapshot code)
  ch09/                 # Multi-Agent Systems (+ chapter snapshot code)
  ch10/                 # Evaluation
```

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up LLM provider and API keys in .env
cp .env.example .env
# Edit .env and add your API keys

# Launch Jupyter Lab
uv run jupyter lab
```

## LLM provider and API keys

Create a `.env` file in the project root with the following keys:

```
LLM_PROVIDER=ollama            # Optional; Ollama is the default
OLLAMA_NETWORK_HOST=http://host:11434
OLLAMA_DEFAULT_MODEL=qwen3:8b

OPENAI_API_KEY=sk-...          # Required only when using OpenAI
OPENAI_DEFAULT_MODEL=gpt-5-mini
ANTHROPIC_API_KEY=sk-ant-...   # Required for CH02 Anthropic examples
TAVILY_API_KEY=tvly-...        # Required for CH03 web search
HF_TOKEN=hf_...                # Required for CH02 GAIA benchmark
E2B_API_KEY=e2b_...            # Required for CH08 code execution
```

`LlmClient()` uses Ollama by default. You can also select the provider in code:

```python
ollama = LlmClient()  # model and host come from .env
openai = LlmClient(model="gpt-5-mini", provider="openai")
```

Set `LLM_PROVIDER=openai` to make OpenAI the environment-wide default. An
`OPENAI_API_KEY` is only needed for OpenAI calls.

Run the agent as a script with a required query:

```bash
uv run python -m src.agent "What is 2 + 2?"
uv run python -m src.agent "Summarize this topic" --model qwen3:8b
uv run python -m src.agent "What is 2 + 2?" --provider openai --model gpt-5-mini
```

## Agent Workbench GUI

Launch the Streamlit workbench from the project root:

```bash
uv run streamlit run streamlit_app.py
```

The workbench provides:

- Ollama/OpenAI provider and model selection, including discovery of models
  installed on the configured Ollama server.
- Sampling, token, timeout, seed, system-instruction, and maximum-step controls.
- Per-run tool selection and a browsable tool-schema catalog.
- Persistent chat-style results for the current browser session.
- A detailed execution timeline containing model messages, tool parameters,
  tool results, timing, token usage, raw event payloads, and JSON export.

## Docker deployment

Build and launch the complete Streamlit application with one command:

```bash
./scripts/deploy.sh
```

The script builds the image, replaces the existing `agent-workbench`
container, waits for its health check, and exposes it at
`http://localhost:8501`. Runtime credentials and provider settings are read
from `.env`, which is never copied into the image.

Common overrides:

```bash
PORT=8080 ./scripts/deploy.sh
./scripts/deploy.sh --image ghcr.io/owner/agent --tag latest --pull
./scripts/deploy.sh --help
```

### GitHub Actions CI/CD

[`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml) performs the
following pipeline:

1. Pull requests and pushes targeting `dev` or `main` run Ruff and the full
   pytest suite using the locked environment.
2. A successful push to `main` (including a merged PR) builds the Docker image
   and publishes `sha-<commit>` and `latest` tags to GitHub Container Registry.
3. The immutable SHA image is pulled onto the production Docker host and
   deployed with `scripts/deploy.sh`.

Create a protected GitHub environment named `production` and configure these
repository/environment secrets:

| Name | Purpose |
| --- | --- |
| `DEPLOY_HOST` | DNS name or IP address of the Docker host |
| `DEPLOY_USER` | SSH user with permission to run Docker |
| `DEPLOY_SSH_KEY` | Private SSH key for that user |
| `DEPLOY_PORT` | SSH port; optional, defaults to `22` |
| `DEPLOY_ENV_FILE` | Remote runtime env file; optional, defaults to `/opt/agent/.env` |
| `GHCR_USERNAME` | GitHub user or service account used by the remote host |
| `GHCR_TOKEN` | Token with `read:packages` permission |

Optionally set the `APP_PORT` repository variable to change the production
host port from `8501`. The deployment host needs Docker and an environment
file containing the appropriate OpenAI/Ollama and tool credentials.

## Chapters

| Chapter | Topic | Key Modules |
|---------|-------|-------------|
| CH02 | LLM API Basics | eval/gaia.py |
| CH03 | Tools and Function Calling | tools/helpers.py, tools/calculator.py, tools/search.py |
| CH04 | ReAct Agent | types.py, context.py, llm.py, agent.py, tools/base.py |
| CH05 | RAG and File Tools | rag.py, callbacks.py, tools/file_tools.py |
| CH06 | Memory Systems | memory/session.py, memory/long_term.py, memory/context_optimizer.py |
| CH07 | Planning and Reflection | planning.py |
| CH08 | Code Execution | tools/code_execution.py, skills.py |
| CH09 | Multi-Agent Systems | workflows/, transfer.py, tools/agent_tool.py |
| CH10 | Evaluation | eval/prompts.py |

## Chapter Snapshot Files

Some notebook directories (ch04, ch05, ch06, ch08, ch09) contain `.py` snapshot files that represent the state of core modules *at that chapter*. This lets each chapter's notebook use the version of the code that matches what has been introduced so far, without exposing features from later chapters.
