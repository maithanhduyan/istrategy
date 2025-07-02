import contextlib
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP
from typing import Dict
import uvicorn

# Import tools modules
from .tools import echo_tool, time_tool, calculator_tools, qdrant_tools, embedding_tools

# Create MCP server - sử dụng stateless mode cho streamable HTTP
mcp = FastMCP("qdrant-mcp", stateless_http=True)

# Cấu hình port
mcp.settings.port = 3002
mcp.settings.host = "0.0.0.0"

# Register all tools
echo_tool(mcp)
time_tool(mcp)
calculator_tools(mcp)
qdrant_tools(mcp)
embedding_tools(mcp)

# Legacy tools for compatibility
@mcp.tool()
async def get_server_status() -> Dict:
    """Get the current status of the MCP server."""
    return {
        "status": "running",
        "server_name": "qdrant-mcp",
        "tools_available": [
            # Echo tools
            "echo",
            # Time tools
            "get_current_time", "get_timestamp", "format_timestamp",
            # Calculator tools
            "add", "subtract", "multiply", "divide", "power", "square_root", "calculate_expression",
            # Qdrant tools
            "qdrant_status", "list_collections", "create_collection", 
            "search_vectors", "insert_vectors", "get_collection_info",
            # Embedding tools
            "create_embedding", "create_batch_embeddings", "get_embedding_model_info",
            "list_embedding_models", "embed_and_store_in_qdrant",
            # Legacy
            "get_server_status"
        ],
        "transport": "streamable-http",
        "endpoint": "/mcp"
    }

def main():
    """Entry point for the Qdrant MCP server."""
    print("Starting Qdrant MCP server with streamable-http transport")
    print("Server will be available at http://localhost:3002/mcp")
    
    # Run MCP server trực tiếp với streamable-http transport
    # Server sẽ tự động expose endpoint tại /mcp
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
