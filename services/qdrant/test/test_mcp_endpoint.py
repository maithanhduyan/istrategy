#!/usr/bin/env python3
"""Test MCP client for the /mcp endpoint."""

import asyncio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def test_mcp_endpoint():
    """Test MCP tools via streamable HTTP."""
    try:
        # Connect to streamable HTTP server tại /mcp/ endpoint
        async with streamablehttp_client("http://localhost:3002/mcp/") as (
            read_stream,
            write_stream,
            _,
        ):
            # Create MCP session
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize connection
                print("Initializing MCP connection...")
                await session.initialize()
                print("✓ Connection initialized")

                # List available tools
                print("\nListing tools...")
                tools_response = await session.list_tools()
                print(f"✓ Found {len(tools_response.tools)} tools:")
                for tool in tools_response.tools:
                    print(f"  - {tool.name}: {tool.description}")

                # Test echo tool
                print("\nTesting echo tool...")
                echo_result = await session.call_tool("echo", {"message": "Hello from test!"})
                print(f"✓ Echo result: {echo_result.content[0].text}")

                # Test add tool
                print("\nTesting add tool...")
                add_result = await session.call_tool("add", {"a": 5, "b": 3})
                print(f"✓ Add result: {add_result.content[0].text}")

                # Test get_server_status tool
                print("\nTesting get_server_status tool...")
                status_result = await session.call_tool("get_server_status", {})
                print(f"✓ Server status: {status_result.content[0].text}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_endpoint())
    exit(0 if success else 1)
