import asyncio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def test_mcp_connection():
    """Test kết nối đến MCP server và gọi các tools."""
    try:
        # Kết nối đến MCP server với đúng URL
        async with streamablehttp_client("http://localhost:8000/") as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize connection
                await session.initialize()
                print("✅ Connected to MCP server successfully!")
                
                # List available tools
                tools_response = await session.list_tools()
                print(f"📋 Available tools: {[tool.name for tool in tools_response.tools]}")
                
                # Test echo tool
                echo_result = await session.call_tool("echo", {"message": "Hello from MCP client!"})
                print(f"🔄 Echo result: {echo_result.content}")
                
                # Test add tool
                add_result = await session.call_tool("add", {"a": 10, "b": 5})
                print(f"➕ Add result: {add_result.content}")
                
                # Test server status
                status_result = await session.call_tool("get_server_status", {})
                print(f"📊 Server status: {status_result.content}")
                
    except Exception as e:
        print(f"❌ Error connecting to MCP server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
