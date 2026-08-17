import argparse
import asyncio
import logging
import sys
from typing import Any, Sequence  # noqa: UP035

from dotenv import load_dotenv

from src.agent import Agent
from src.calculator import calculator

# Project Im
from src.llm import LlmClient
from src.tools import search_web

# --------------------------------------------------------------------------- #
# Environment setup and Logging
# --------------------------------------------------------------------------- #

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Run Agent
# --------------------------------------------------------------------------- #

def run_agent(client: LlmClient, query: str,
              tools: list | None = None) -> Any:
    """
    Run the agent for the given query and return the result.
    Args:
        client (LlmClient): The LLM client to use for the agent.
        query (str): The user query to process.
        tools (list, optional): A list of tools available to the agent.
        Defaults to None.
    Returns:
        Any: The result of the agent's processing of the query.
    """
    agent = Agent(
        model=client,
        tools=[calculator, search_web],
        instructions="You are a helpful assistant.",
    )

    try:
        result = asyncio.run(agent.run(user_input=query))

    except RuntimeError as error:
        sys.exit(f"error: {error}\n")

    if result.output is not None:
        print(result.output)
    else:
        print("No output or result returned.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Run the agent from the command line."""
    parser = argparse.ArgumentParser(
        description="Run a query through the tool-calling agent.",
    )
    parser.add_argument("query", help="Query to send to the agent")
    parser.add_argument(
        "--model",
        help=(
            "Model name. Defaults to OLLAMA_DEFAULT_MODEL for Ollama or "
            "OPENAI_DEFAULT_MODEL for OpenAI."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        help="LLM provider. Defaults to LLM_PROVIDER, or Ollama when unset.",
    )
    args = parser.parse_args(argv)

    query = args.query
    provider, model = args.provider, args.model
    client = LlmClient(model=model, provider=provider)
    tools = [calculator, search_web]

    result = run_agent(client, query, tools=tools)

    print("Agent finished. Result:", result)

    return result


if __name__ == "__main__":
    main()
