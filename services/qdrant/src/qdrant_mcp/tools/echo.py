"""Echo tool for MCP server."""

from mcp.server.fastmcp import FastMCP


def register_echo_tool(mcp_server: FastMCP):
    """Register echo tool with MCP server."""
    
    @mcp_server.tool()
    async def echo(message: str) -> str:
        """Echo back the provided message."""
        return f"Echo: {message}"


# Alias để dễ import
echo_tool = register_echo_tool
