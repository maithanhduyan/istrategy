"""Time tools for MCP server."""

from mcp.server.fastmcp import FastMCP
from datetime import datetime
from typing import Dict
import time


def register_time_tools(mcp_server: FastMCP):
    """Register time-related tools with MCP server."""
    
    @mcp_server.tool()
    async def get_current_time() -> Dict:
        """Get current date and time information."""
        now = datetime.now()
        return {
            "current_time": now.isoformat(),
            "timestamp": int(time.time()),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": str(now.astimezone().tzinfo)
        }
    
    @mcp_server.tool()
    async def get_timestamp() -> int:
        """Get current Unix timestamp."""
        return int(time.time())
    
    @mcp_server.tool()
    async def format_timestamp(timestamp: int, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format Unix timestamp to human readable format."""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime(format_string)


# Alias để dễ import
time_tool = register_time_tools
