import contextlib
import logging
import json
import mcp.types as types
from collections.abc import AsyncIterator
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from typing import Dict
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# Import tools for definitions only
from .tools import get_tool_definitions

# Create MCP server using low-level Server for StreamableHTTP compatibility
app = Server("chromadb-mcp")

# Unified tool handler
@app.call_tool()
async def handle_all_tools(name: str, arguments: dict) -> list[types.ContentBlock]:
    """Handle all tool calls in one unified handler."""
    # Handle server status tool
    if name == "get_server_status":
        tool_definitions = get_tool_definitions()
        tool_names = [tool.name for tool in tool_definitions] + ["get_server_status"]
        
        return [
            types.TextContent(
                type="text", 
                text=str({
                    "status": "running",
                    "server_name": "chromadb-mcp",
                    "tools_available": tool_names,
                    "transport": "streamable-http",
                    "endpoint": "/mcp"
                })
            )
        ]
    
    # Handle echo tool
    elif name == "echo":
        message = arguments.get("message", "")
        return [types.TextContent(type="text", text=f"Echo: {message}")]
    
    # Handle ChromaDB tools
    else:
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            
            if name == "list_collections":
                collections = client.list_collections()
                collection_names = [col.name for col in collections]
                return [types.TextContent(
                    type="text", 
                    text=f"Collections: {json.dumps(collection_names, indent=2)}"
                )]
            
            elif name == "create_collection":
                collection_name = arguments.get("name")
                if not collection_name:
                    return [types.TextContent(type="text", text="Error: collection name is required")]
                
                collection = client.create_collection(name=collection_name)
                return [types.TextContent(
                    type="text", 
                    text=f"Created collection: {collection_name}"
                )]
            
            elif name == "get_collection_info":
                collection_name = arguments.get("name")
                if not collection_name:
                    return [types.TextContent(type="text", text="Error: collection name is required")]
                
                try:
                    collection = client.get_collection(name=collection_name)
                    count = collection.count()
                    return [types.TextContent(
                        type="text", 
                        text=f"Collection '{collection_name}' has {count} documents"
                    )]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
            # Add other tools as needed...
            else:
                return [types.TextContent(type="text", text=f"Tool '{name}' not implemented yet")]
        
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    tools = get_tool_definitions()
    
    # Add server status tool
    tools.append(
        types.Tool(
            name="get_server_status",
            description="Get the current status of the MCP server.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        )
    )
    
    return tools

def main():
    """Entry point for the ChromaDB MCP server with stateless uvicorn integration."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    print("Starting ChromaDB MCP server with streamable-http transport")
    print("Server will be available at http://localhost:3003/mcp")
    
    # Create session manager with stateless mode for better performance
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,  # No event store for stateless mode
        json_response=False,  # Use SSE streams
        stateless=True,  # Enable stateless mode
    )
    
    # ASGI handler for streamable HTTP connections
    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)
    
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        async with session_manager.run():
            logger.info("ChromaDB MCP server started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logger.info("ChromaDB MCP server shutting down...")
    
    # Create Starlette ASGI application
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )
    
    # Run with uvicorn
    try:
        logger.info("Starting uvicorn server with stateless MCP integration...")
        uvicorn.run(starlette_app, host="0.0.0.0", port=3003, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()