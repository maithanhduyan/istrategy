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
    
    # Handle ChromaDB Collection Tools
    elif name == "chroma_list_collections":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            limit = arguments.get("limit")
            offset = arguments.get("offset")
            
            colls = client.list_collections(limit=limit, offset=offset)
            if not colls:
                return [types.TextContent(type="text", text="Collections: []")]
            
            collection_names = [coll.name for coll in colls]
            return [types.TextContent(
                type="text", 
                text=f"Collections: {json.dumps(collection_names, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_create_collection":
        from .tools import get_chroma_client, mcp_known_embedding_functions
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            embedding_function_name = arguments.get("embedding_function_name", "default")
            metadata = arguments.get("metadata")
            
            if not collection_name:
                return [types.TextContent(type="text", text="Error: collection_name is required")]
            
            if embedding_function_name not in mcp_known_embedding_functions:
                return [types.TextContent(
                    type="text", 
                    text=f"Error: Unknown embedding function '{embedding_function_name}'. Available: {list(mcp_known_embedding_functions.keys())}"
                )]
            
            embedding_function = mcp_known_embedding_functions[embedding_function_name]
            
            from chromadb.api.collection_configuration import CreateCollectionConfiguration
            configuration = CreateCollectionConfiguration(
                embedding_function=embedding_function()
            )
            
            client.create_collection(
                name=collection_name,
                configuration=configuration,
                metadata=metadata
            )
            
            return [types.TextContent(
                type="text", 
                text=f"Successfully created collection '{collection_name}' with embedding function '{embedding_function_name}'"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_peek_collection":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            limit = arguments.get("limit", 5)
            
            if not collection_name:
                return [types.TextContent(type="text", text="Error: collection_name is required")]
            
            collection = client.get_collection(collection_name)
            results = collection.peek(limit=limit)
            
            return [types.TextContent(
                type="text", 
                text=f"Peek results:\n{json.dumps(results, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_get_collection_info":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            
            if not collection_name:
                return [types.TextContent(type="text", text="Error: collection_name is required")]
            
            collection = client.get_collection(collection_name)
            count = collection.count()
            # Không include peek results để tránh numpy serialization issues
            
            info = {
                "name": collection_name,
                "count": count
            }
            
            return [types.TextContent(
                type="text", 
                text=f"Collection info:\n{json.dumps(info, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_get_collection_count":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            
            if not collection_name:
                return [types.TextContent(type="text", text="Error: collection_name is required")]
            
            collection = client.get_collection(collection_name)
            count = collection.count()
            
            return [types.TextContent(
                type="text", 
                text=f"Collection '{collection_name}' has {count} documents"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_modify_collection":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            new_name = arguments.get("new_name")
            new_metadata = arguments.get("new_metadata")
            
            if not collection_name:
                return [types.TextContent(type="text", text="Error: collection_name is required")]
            
            collection = client.get_collection(collection_name)
            collection.modify(name=new_name, metadata=new_metadata)
            
            modified_aspects = []
            if new_name:
                modified_aspects.append("name")
            if new_metadata:
                modified_aspects.append("metadata")
            
            return [types.TextContent(
                type="text", 
                text=f"Successfully modified collection '{collection_name}': updated {' and '.join(modified_aspects)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_delete_collection":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            
            if not collection_name:
                return [types.TextContent(type="text", text="Error: collection_name is required")]
            
            client.delete_collection(collection_name)
            return [types.TextContent(
                type="text", 
                text=f"Successfully deleted collection '{collection_name}'"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
    # Handle ChromaDB Document Tools
    elif name == "chroma_add_documents":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            documents = arguments.get("documents", [])
            ids = arguments.get("ids", [])
            metadatas = arguments.get("metadatas")
            
            if not collection_name or not documents or not ids:
                return [types.TextContent(
                    type="text", 
                    text="Error: collection_name, documents, and ids are required"
                )]
            
            if len(ids) != len(documents):
                return [types.TextContent(
                    type="text", 
                    text=f"Error: Number of ids ({len(ids)}) must match number of documents ({len(documents)})"
                )]
            
            # Check for empty IDs
            if any(not id.strip() for id in ids):
                return [types.TextContent(type="text", text="Error: IDs cannot be empty strings")]
            
            collection = client.get_or_create_collection(collection_name)
            
            # Check for duplicate IDs
            existing_ids = collection.get(include=[])["ids"]
            duplicate_ids = [id for id in ids if id in existing_ids]
            
            if duplicate_ids:
                return [types.TextContent(
                    type="text", 
                    text=f"Error: The following IDs already exist in collection '{collection_name}': {duplicate_ids}. Use 'chroma_update_documents' to update existing documents."
                )]
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return [types.TextContent(
                type="text", 
                text=f"Successfully added {len(documents)} documents to collection '{collection_name}'"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_query_collection":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            query_texts = arguments.get("query_texts", [])
            n_results = arguments.get("n_results", 5)
            where = arguments.get("where")
            
            if not collection_name or not query_texts:
                return [types.TextContent(
                    type="text", 
                    text="Error: collection_name and query_texts are required"
                )]
            
            collection = client.get_collection(collection_name)
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            
            return [types.TextContent(
                type="text", 
                text=f"Query results:\n{json.dumps(results, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_update_documents":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            ids = arguments.get("ids", [])
            documents = arguments.get("documents")
            metadatas = arguments.get("metadatas")
            
            if not collection_name or not ids:
                return [types.TextContent(
                    type="text", 
                    text="Error: collection_name and ids are required"
                )]
            
            collection = client.get_collection(collection_name)
            collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            return [types.TextContent(
                type="text", 
                text=f"Successfully updated {len(ids)} documents in collection '{collection_name}'"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_get_documents":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            ids = arguments.get("ids")
            where = arguments.get("where")
            limit = arguments.get("limit")
            offset = arguments.get("offset")
            include = arguments.get("include", ["documents", "metadatas"])
            
            if not collection_name:
                return [types.TextContent(
                    type="text", 
                    text="Error: collection_name is required"
                )]
            
            collection = client.get_collection(collection_name)
            results = collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=include
            )
            
            return [types.TextContent(
                type="text", 
                text=f"Documents from '{collection_name}':\n{json.dumps(results, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "chroma_delete_documents":
        from .tools import get_chroma_client
        try:
            client = get_chroma_client()
            collection_name = arguments.get("collection_name")
            ids = arguments.get("ids")
            where = arguments.get("where")
            
            if not collection_name or (not ids and not where):
                return [types.TextContent(
                    type="text", 
                    text="Error: collection_name and either ids or where condition are required"
                )]
            
            collection = client.get_collection(collection_name)
            collection.delete(
                ids=ids,
                where=where
            )
            
            return [types.TextContent(
                type="text", 
                text=f"Successfully deleted documents from collection '{collection_name}'"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

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