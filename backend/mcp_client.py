"""
MCP Client for Health Agent
===========================
This module provides functions to call MCP tools from the LangGraph agent.

How MCP Client-Server Communication Works:
1. The client spawns the MCP server as a subprocess
2. They communicate over stdio (standard input/output)
3. The client sends tool call requests, server executes and returns results
4. This is the standard MCP transport mechanism

Usage in the agent:
    from mcp_client import call_mcp_tool
    result = await call_mcp_tool("log_triage_result", {...})
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Server script path - points to our MCP server
SERVER_SCRIPT = "mcp_server.py"


@asynccontextmanager
async def get_mcp_client():
    """
    Create a connection to the MCP server.

    This context manager:
    1. Spawns mcp_server.py as a subprocess
    2. Establishes stdio communication
    3. Initializes the MCP session
    4. Yields the session for tool calls
    5. Cleans up when done
    """
    # Configure how to launch the server
    server_params = StdioServerParameters(
        command=sys.executable,  # Use the same Python interpreter
        args=[SERVER_SCRIPT],    # Run our server script
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection (required handshake)
            await session.initialize()
            yield session


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Call a tool on the MCP server.

    Args:
        tool_name: Name of the tool to call (e.g., "log_triage_result")
        arguments: Dictionary of arguments to pass to the tool

    Returns:
        The tool's response as a dictionary

    Example:
        result = await call_mcp_tool("log_triage_result", {
            "patient_message": "I have a headache",
            "intake_data": {"severity": "mild"},
            "triage_decision": "non-urgent",
            "is_urgent": False,
            "ai_response": "Based on your symptoms..."
        })
    """
    async with get_mcp_client() as session:
        # Call the tool and get the result
        result = await session.call_tool(tool_name, arguments)

        # Extract the content from the result
        # MCP returns content as a list of content blocks
        if result.content:
            # Get the text content from the first block
            content = result.content[0]
            if hasattr(content, 'text'):
                import json
                try:
                    return json.loads(content.text)
                except json.JSONDecodeError:
                    return {"text": content.text}
        return {"success": True}


async def list_available_tools():
    """
    List all tools available on the MCP server.
    Useful for debugging and verification.
    """
    async with get_mcp_client() as session:
        tools = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
            for tool in tools.tools
        ]


# =============================================================================
# Synchronous wrappers for use in LangGraph
# =============================================================================
# LangGraph nodes are typically synchronous, so we provide sync wrappers
# that run the async functions in an event loop.

def call_tool_sync(tool_name: str, arguments: dict) -> dict:
    """
    Synchronous wrapper for call_mcp_tool.
    Use this in LangGraph nodes.
    """
    return asyncio.run(call_mcp_tool(tool_name, arguments))


def list_tools_sync() -> list:
    """
    Synchronous wrapper for list_available_tools.
    """
    return asyncio.run(list_available_tools())


# =============================================================================
# Test the client
# =============================================================================
if __name__ == "__main__":
    print("Testing MCP Client...")
    print("-" * 50)

    # List available tools
    print("\nAvailable tools:")
    tools = list_tools_sync()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")

    # Test log_triage_result
    print("\nTesting log_triage_result...")
    result = call_tool_sync("log_triage_result", {
        "patient_message": "I have a headache",
        "intake_data": {"severity": "mild", "fever": "no"},
        "triage_decision": "non-urgent",
        "is_urgent": False,
        "ai_response": "Based on your symptoms, this appears to be a tension headache."
    })
    print(f"Result: {result}")

    print("\nMCP Client test complete!")