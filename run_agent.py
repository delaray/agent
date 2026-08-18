import argparse  # noqa: I001
import asyncio
import logging
import sys
import pprint
from typing import Any, Sequence  # noqa: UP035

from dotenv import load_dotenv

from src.utils import init_logging
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
logger = init_logging(log_dir="logs", log_file="agent.log")


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
        tools=tools,
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

    return result


# --------------------------------------------------------------------------- #
# CLI Argumernts
# --------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(
    description="Run a query through the tool-calling agent.",
)

parser.add_argument("query", help="Query to send to the agent")

parser.add_argument("--model",
                    default="qwen3.8:27b",
                    help="Model name. Defaults to qwen3.8:27b "
                    )

parser.add_argument("--provider", choices=("ollama", "openai"),
                    default="ollama",
                    help="LLM provider. Defaults to ollama.",
                    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Run the agent from the command line."""
    args = parser.parse_args(argv)

    query = args.query
    provider, model = args.provider, args.model
    client = LlmClient(model=model, provider=provider)
    tools = [calculator, search_web]

    result = run_agent(client, query, tools=tools)

    print("\n\nAgent finished.\nResult:")
    print(result)

    return result


if __name__ == "__main__":
    main()
