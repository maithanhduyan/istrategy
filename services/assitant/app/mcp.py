import json
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.types import Receive, Scope, Send
from starlette.responses import Response
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List
from app.auth import get_current_user, verify_mcp_api_key
from app.db import get_user_by_username
from app.pipeline import (
    get_collections, create_collection_with_embeddings,
    add_text_documents_with_embeddings, search_similar_documents,
    get_collection_summary, batch_process_texts
)

router = APIRouter(dependencies=[Depends(verify_mcp_api_key)])

# Create MCP server using low-level Server for StreamableHTTP compatibility
mcp_app = Server("assistant-mcp")

async def server_status() -> str:
    """Check the server status."""
    return "Server is running smoothly."

async def get_database_info() -> str:
    """Get database information."""
    return "SQLite database with users table initialized successfully."

async def get_user_info(username: str) -> str:
    """Get user information by username."""
    user = get_user_by_username(username)
    if user:
        return f"User found: {user['username']} (ID: {user['id']}, Created: {user['created_at']})"
    return f"User '{username}' not found."

async def get_chroma_collections() -> str:
    """Get list of ChromaDB collections."""
    try:
        collections = await get_collections()
        if collections:
            return f"ChromaDB collections: {', '.join(collections)}"
        else:
            return "No ChromaDB collections found."
    except Exception as e:
        return f"Error retrieving ChromaDB collections: {str(e)}"

async def create_collection(name: str, description: str = "") -> str:
    """Create a new ChromaDB collection with embeddings."""
    try:
        metadata = {"description": description or f"Collection {name}"}
        result = await create_collection_with_embeddings(name, metadata)
        return f"Created collection '{name}' successfully. Status: {result['status']}"
    except Exception as e:
        return f"Error creating collection '{name}': {str(e)}"

async def add_documents(collection_name: str, texts: str) -> str:
    """Add documents to a ChromaDB collection."""
    try:
        # Split texts by newlines or semicolons
        text_list = [t.strip() for t in texts.replace('\n', ';').split(';') if t.strip()]
        result = await add_text_documents_with_embeddings(collection_name, text_list)
        return f"Added {result['documents_added']} documents to '{collection_name}'"
    except Exception as e:
        return f"Error adding documents to '{collection_name}': {str(e)}"

async def search_documents(collection_name: str, query: str, n_results: int = 5) -> str:
    """Search for similar documents in a ChromaDB collection."""
    try:
        result = await search_similar_documents(collection_name, query, n_results)
        docs = result['results']['documents'][0] if result['results']['documents'] else []
        distances = result['results']['distances'][0] if result['results']['distances'] else []
        
        response = f"Found {len(docs)} documents for query '{query}':\n"
        for i, (doc, dist) in enumerate(zip(docs, distances)):
            response += f"{i+1}. (distance: {dist:.3f}) {doc[:100]}...\n"
        return response
    except Exception as e:
        return f"Error searching in '{collection_name}': {str(e)}"

async def get_collection_info(collection_name: str) -> str:
    """Get comprehensive information about a ChromaDB collection."""
    try:
        summary = await get_collection_summary(collection_name)
        stats = summary['stats']['stats']
        return f"Collection '{collection_name}': {stats['count']} documents, metadata: {stats.get('metadata', {})}"
    except Exception as e:
        return f"Error getting info for '{collection_name}': {str(e)}"

# MCP Tool Handlers
@mcp_app.call_tool()
async def handle_all_tools(name: str, arguments: dict) -> list[types.ContentBlock]:
    """Handle all tool calls in one unified handler."""
    match name:
        case "server_status":
            status = await server_status()
            return [types.ContentBlock(type="text", text=status)]
        case "database_info":
            info = await get_database_info()
            return [types.ContentBlock(type="text", text=info)]
        case "user_info":
            username = arguments.get("username", "")
            info = await get_user_info(username)
            return [types.ContentBlock(type="text", text=info)]
        case "chroma_get_collections":
            collections_info = await get_chroma_collections()
            return [types.ContentBlock(type="text", text=collections_info)]
        case "chroma_create_collection":
            name = arguments.get("name", "")
            description = arguments.get("description", "")
            result = await create_collection(name, description)
            return [types.ContentBlock(type="text", text=result)]
        case "chroma_add_documents":
            collection_name = arguments.get("collection_name", "")
            texts = arguments.get("texts", "")
            result = await add_documents(collection_name, texts)
            return [types.ContentBlock(type="text", text=result)]
        case "chroma_search_documents":
            collection_name = arguments.get("collection_name", "")
            query = arguments.get("query", "")
            n_results = arguments.get("n_results", 5)
            result = await search_documents(collection_name, query, n_results)
            return [types.ContentBlock(type="text", text=result)]
        case "chroma_collection_info":
            collection_name = arguments.get("collection_name", "")
            result = await get_collection_info(collection_name)
            return [types.ContentBlock(type="text", text=result)]
        case _:
            raise ValueError(f"Unknown tool: {name}")

# MCP List Tools
@mcp_app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    tools = [
        types.Tool(
            name="server_status",
            description="Get the current status of the MCP server.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="database_info",
            description="Get information about the database.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="user_info",
            description="Get information about a specific user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username to look up"
                    }
                },
                "required": ["username"]
            },
        ),
        types.Tool(
            name="chroma_get_collections",
            description="Get the list of ChromaDB collections.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="chroma_create_collection",
            description="Create a new ChromaDB collection with embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection to create"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description for the collection"
                    }
                },
                "required": ["name"]
            },
        ),
        types.Tool(
            name="chroma_add_documents",
            description="Add text documents to a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection"
                    },
                    "texts": {
                        "type": "string",
                        "description": "Text documents separated by semicolons or newlines"
                    }
                },
                "required": ["collection_name", "texts"]
            },
        ),
        types.Tool(
            name="chroma_search_documents",
            description="Search for similar documents in a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to search"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["collection_name", "query"]
            },
        ),
        types.Tool(
            name="chroma_collection_info",
            description="Get information about a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection"
                    }
                },
                "required": ["collection_name"]
            },
        ),
    ]
    return tools

# Create session manager with stateless mode for better performance
session_manager = StreamableHTTPSessionManager(
    app=mcp_app,
    event_store=None,  # No event store for stateless mode
    json_response=True,  # Use JSON for VS Code compatibility
    stateless=False,  # Disable stateless for proper MCP protocol
)

# Main MCP endpoint for VS Code integration  
@router.api_route("/", methods=["GET", "POST"])
@router.api_route("", methods=["GET", "POST"])
async def handle_mcp_protocol(request: Request):
    """Handle MCP protocol requests from VS Code."""
    try:
        # Simple MCP protocol implementation for VS Code
        if request.method == "GET":
            # Return server info for GET requests
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "logging": {}
                    },
                    "serverInfo": {
                        "name": "assistant-mcp",
                        "version": "1.0.0"
                    }
                }
            }
        
        elif request.method == "POST":
            body = await request.json()
            
            # Handle initialize request
            if body.get("method") == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "logging": {}
                        },
                        "serverInfo": {
                            "name": "assistant-mcp",
                            "version": "1.0.0"
                        }
                    }
                }
            
            # Handle tools/list request
            elif body.get("method") == "tools/list":
                tools = await list_tools()
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in tools
                        ]
                    }
                }
            
            # Handle tools/call request
            elif body.get("method") == "tools/call":
                params = body.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                try:
                    # Direct tool execution without MCP decorator
                    match tool_name:
                        case "server_status":
                            result_text = await server_status()
                        case "database_info":
                            result_text = await get_database_info()
                        case "user_info":
                            username = arguments.get("username", "")
                            result_text = await get_user_info(username)
                        case "chroma_get_collections":
                            result_text = await get_chroma_collections()
                        case "chroma_create_collection":
                            name = arguments.get("name", "")
                            description = arguments.get("description", "")
                            result_text = await create_collection(name, description)
                        case "chroma_add_documents":
                            collection_name = arguments.get("collection_name", "")
                            texts = arguments.get("texts", "")
                            result_text = await add_documents(collection_name, texts)
                        case "chroma_search_documents":
                            collection_name = arguments.get("collection_name", "")
                            query = arguments.get("query", "")
                            n_results = arguments.get("n_results", 5)
                            result_text = await search_documents(collection_name, query, n_results)
                        case "chroma_collection_info":
                            collection_name = arguments.get("collection_name", "")
                            result_text = await get_collection_info(collection_name)
                        case _:
                            raise ValueError(f"Unknown tool: {tool_name}")
                    
                    return {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result_text
                                }
                            ]
                        }
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {
                            "code": -32603,
                            "message": str(e)
                        }
                    }
            
            # Handle other methods
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {body.get('method')}"
                    }
                }
        
        return {"error": "Invalid request"}
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

# REST API endpoints for MCP functionality (for testing)
@router.get("/info")
async def mcp_info():
    """Get MCP server information."""
    return {
        "name": "assistant-mcp",
        "version": "1.0.0",
        "description": "Model Context Protocol server for Assistant service",
        "status": "running",
        "features": [
            "Authentication integration",
            "Database operations", 
            "Server monitoring"
        ],
        "security": "API Key protected"
    }

@router.get("/tools")
async def get_mcp_tools():
    """Get available MCP tools."""
    tools = await list_tools()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in tools
        ]
    }

@router.post("/tools/{tool_name}")
async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any] = None):
    """Call an MCP tool."""
    if arguments is None:
        arguments = {}
    
    try:
        # Direct tool execution without MCP server wrapper
        match tool_name:
            case "server_status" | "get_server_status":
                status = await server_status()
                result = [{"type": "text", "content": status}]
            case "database_info" | "get_database_info":
                info = await get_database_info()
                result = [{"type": "text", "content": info}]
            case "user_info" | "get_user_info":
                username = arguments.get("username", "")
                if not username:
                    raise ValueError("Username is required for user_info tool")
                info = await get_user_info(username)
                result = [{"type": "text", "content": info}]
            case "chroma_get_collections" | "get_chroma_collections":
                collections_info = await get_chroma_collections()
                result = [{"type": "text", "content": collections_info}]
            case "chroma_create_collection" | "create_collection":
                name = arguments.get("name", "")
                description = arguments.get("description", "")
                if not name:
                    raise ValueError("Collection name is required")
                collections_info = await create_collection(name, description)
                result = [{"type": "text", "content": collections_info}]
            case "chroma_add_documents" | "add_documents":
                collection_name = arguments.get("collection_name", "")
                texts = arguments.get("texts", "")
                if not collection_name or not texts:
                    raise ValueError("Collection name and texts are required")
                collections_info = await add_documents(collection_name, texts)
                result = [{"type": "text", "content": collections_info}]
            case "chroma_search_documents" | "search_documents":
                collection_name = arguments.get("collection_name", "")
                query = arguments.get("query", "")
                n_results = arguments.get("n_results", 5)
                if not collection_name or not query:
                    raise ValueError("Collection name and query are required")
                collections_info = await search_documents(collection_name, query, n_results)
                result = [{"type": "text", "content": collections_info}]
            case "chroma_collection_info" | "collection_info":
                collection_name = arguments.get("collection_name", "")
                if not collection_name:
                    raise ValueError("Collection name is required")
                collections_info = await get_collection_info(collection_name)
                result = [{"type": "text", "content": collections_info}]
            case _:
                raise ValueError(f"Unknown tool: {tool_name}")
        
        return {
            "tool": tool_name,
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# Protected MCP endpoints that require authentication
@router.get("/protected/current-user")
async def get_current_user_mcp(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user via MCP."""
    return {
        "mcp_tool": "current_user",
        "result": {
            "type": "user_info",
            "content": current_user
        }
    }