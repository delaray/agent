import asyncio  # noqa: I001
import os
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv(override=True)


def tavily_server_params():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
        }
    )
    return server_params


# -----------------------------------------------------------------------------
# List Tavily MCP Client Tools
# -----------------------------------------------------------------------------

async def list_tavily_tools(server_params):
    async with stdio_client(server_params) as (read_stream, write_stream), \
               ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        # List available tools
        tools_result = await session.list_tools()
        print("Available tools:")
        for tool in tools_result.tools:
            print(f" - {tool.name}: {tool.description}")


# -----------------------------------------------------------------------------
# Run Tavily MCP Client Query
# -----------------------------------------------------------------------------

async def run_tavily_query(query, server_params):
    async with stdio_client(server_params) as (read_stream, write_stream), \
               ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        result = await session.call_tool(
            "tavily_search",
            arguments={"query": query}
        )
        return result.content


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    # Get Tavily MCP server parameters
    params = tavily_server_params()

    # List available tools
    tools = asyncio.run(list_tavily_tools(params))
    print(f"Tools: {tools}")

    # Run a query using the Tavily MCP client
    query = "What is the capital of France?"
    result = asyncio.run(run_tavily_query(query, params))
    print(f"Query Result: {result}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# End of File
# -----------------------------------------------------------------------------
