
import mcp.types as types
from mcp.server.lowlevel import Server
import chromadb
from chromadb.config import Settings
import json
import logging
from typing import List, Dict, Any, Optional


# Global variables
_chroma_client = None
logger = logging.getLogger(__name__)

def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.Client(Settings(
                is_persistent=True,
                persist_directory="./chromadb"
            ))
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            # Fallback to in-memory client
            _chroma_client = chromadb.Client()
            logger.info("Using in-memory ChromaDB client as fallback")
    return _chroma_client

def register_tools(server: Server):
    """Register all tools with the MCP server."""
    register_echo_tool(server)
    register_chromadb_tools(server)

def register_echo_tool(server: Server):
    """Register echo tool with MCP server."""
    
    # Echo tool is handled in register_chromadb_tools to avoid conflicts
    pass

def register_chromadb_tools(server: Server):
    """Register ChromaDB tools with MCP server."""
    
    @server.call_tool()
    async def handle_all_tools(name: str, arguments: dict) -> list[types.ContentBlock]:
        try:
            # Handle echo tool
            if name == "echo":
                message = arguments.get("message", "")
                return [types.TextContent(type="text", text=f"Echo: {message}")]
            
            # Handle ChromaDB tools
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
            
            elif name == "add_documents":
                collection_name = arguments.get("collection_name")
                documents = arguments.get("documents", [])
                ids = arguments.get("ids", [])
                metadatas = arguments.get("metadatas", [])
                
                if not collection_name or not documents:
                    return [types.TextContent(
                        type="text", 
                        text="Error: collection_name and documents are required"
                    )]
                
                try:
                    collection = client.get_collection(name=collection_name)
                    
                    # Generate IDs if not provided
                    if not ids:
                        ids = [f"doc_{i}" for i in range(len(documents))]
                    
                    collection.add(
                        documents=documents,
                        ids=ids,
                        metadatas=metadatas if metadatas else None
                    )
                    
                    return [types.TextContent(
                        type="text", 
                        text=f"Added {len(documents)} documents to '{collection_name}'"
                    )]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
            elif name == "query_collection":
                collection_name = arguments.get("collection_name")
                query_texts = arguments.get("query_texts", [])
                n_results = arguments.get("n_results", 5)
                
                if not collection_name or not query_texts:
                    return [types.TextContent(
                        type="text", 
                        text="Error: collection_name and query_texts are required"
                    )]
                
                try:
                    collection = client.get_collection(name=collection_name)
                    results = collection.query(
                        query_texts=query_texts,
                        n_results=n_results
                    )
                    
                    return [types.TextContent(
                        type="text", 
                        text=f"Query results:\n{json.dumps(results, indent=2)}"
                    )]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
            elif name == "delete_collection":
                collection_name = arguments.get("name")
                if not collection_name:
                    return [types.TextContent(type="text", text="Error: collection name is required")]
                
                try:
                    client.delete_collection(name=collection_name)
                    return [types.TextContent(
                        type="text", 
                        text=f"Deleted collection: {collection_name}"
                    )]
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
        except Exception as e:
            logger.error(f"Error in ChromaDB tool '{name}': {e}")
            return [types.TextContent(type="text", text=f"Unexpected error: {str(e)}")]
        
        return []

def get_tool_definitions() -> list[types.Tool]:
    """Get all tool definitions for registration."""
    return [
        # Echo tool
        types.Tool(
            name="echo",
            description="Echo back the provided message.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }
        ),
        # ChromaDB tools
        types.Tool(
            name="list_collections",
            description="List all ChromaDB collections.",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="create_collection",
            description="Create a new ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection to create"
                    }
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get_collection_info",
            description="Get information about a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection"
                    }
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="add_documents",
            description="Add documents to a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection"
                    },
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document texts"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of document IDs"
                    },
                    "metadatas": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional list of metadata objects"
                    }
                },
                "required": ["collection_name", "documents"]
            }
        ),
        types.Tool(
            name="query_collection",
            description="Query documents in a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection"
                    },
                    "query_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of query texts"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["collection_name", "query_texts"]
            }
        ),
        types.Tool(
            name="delete_collection",
            description="Delete a ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the collection to delete"
                    }
                },
                "required": ["name"]
            }
        )
    ]