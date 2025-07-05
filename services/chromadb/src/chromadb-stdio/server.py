# Tích hợp Nomic Embeddings vào ChromaDB thành một pipeline hoàn chỉnh
import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from mcp.server.lowlevel import Server
from typing import Any, List, Dict, Literal, Optional, cast
import json
from mcp import types
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create MCP server using low-level Server for StreamableHTTP compatibility
app = Server("chromadb-mcp")

# Global variables
_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        persist_dir = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def load_embedding_function():
    """Tải hàm embedding từ SentenceTransformer."""
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
    )
    return model.encode


def embedding(text) -> list:
    """Hàm embedding sử dụng SentenceTransformer."""
    model = load_embedding_function()
    return model(text).tolist()


async def chroma_list_collections(
    limit: Optional[int] = None, offset: Optional[int] = None
) -> dict:
    """List all collection names in the Chroma database."""
    client = get_chroma_client()
    try:
        collections = client.list_collections()
        names = [c.name for c in collections]
        if offset:
            names = names[offset:]
        if limit:
            names = names[:limit]
        return {"collections": names}
    except Exception as e:
        return {"error": str(e)}


async def chroma_create_collection(
    collection_name: str,
    embedding_function_name: str = "default",
    metadata: Optional[dict] = None,
) -> dict:
    """Create a new Chroma collection with embedding dimension metadata."""
    client = get_chroma_client()
    try:
        # Lấy dimension từ model embedding
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
        )
        embedding_dim = model.get_sentence_embedding_dimension()
        # Ghi metadata embedding_dim
        if metadata is None:
            metadata = {}
        metadata["embedding_dim"] = embedding_dim
        collection = client.create_collection(name=collection_name, metadata=metadata)
        return {
            "result": f"Collection '{collection_name}' created successfully.",
            "embedding_dim": embedding_dim,
        }
    except Exception as e:
        return {"error": str(e)}


async def chroma_get_collection_info(collection_name: str) -> dict:
    """Get information about a Chroma collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return {
            "name": collection.name,
            "metadata": getattr(collection, "metadata", None),
        }
    except Exception as e:
        return {"error": str(e)}


async def chroma_get_collection_count(collection_name: str) -> dict:
    """Get the number of documents in a Chroma collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return {"count": collection.count()}
    except Exception as e:
        return {"error": str(e)}


async def chroma_modify_collection(
    collection_name: str,
    new_name: Optional[str] = None,
    new_metadata: Optional[dict] = None,
) -> dict:
    """Modify a Chroma collection's name or metadata."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        if new_name:
            collection.modify(name=new_name)
        if new_metadata:
            collection.modify(metadata=new_metadata)
        return {"result": f"Collection '{collection_name}' modified successfully."}
    except Exception as e:
        return {"error": str(e)}


async def chroma_delete_collection(collection_name: str) -> dict:
    """Delete a Chroma collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        return {"result": f"Collection '{collection_name}' deleted successfully."}
    except Exception as e:
        return {"error": str(e)}


async def chroma_add_documents(
    collection_name: str,
    documents: list[str],
    ids: list[str],
    metadatas: Optional[list[dict]] = None,
) -> dict:
    """Add documents to a Chroma collection, always encode embedding with correct dimension."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        # Lấy dimension từ metadata
        embedding_dim = None
        if hasattr(collection, "metadata") and collection.metadata:
            embedding_dim = collection.metadata.get("embedding_dim")
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
        )
        model_dim = model.get_sentence_embedding_dimension()
        if embedding_dim and embedding_dim != model_dim:
            return {
                "error": f"Model embedding dim {model_dim} != collection dim {embedding_dim}"
            }
        # Encode embedding
        embeddings = model.encode(documents)
        if embeddings.shape[1] != model_dim:
            return {
                "error": f"Embedding shape mismatch: {embeddings.shape[1]} != {model_dim}"
            }
        collection.add(
            documents=list(documents),
            ids=list(ids),
            embeddings=embeddings.tolist(),
            metadatas=list(metadatas) if metadatas else None,
        )
        return {
            "result": f"Added {len(documents)} documents to collection '{collection_name}'."
        }
    except Exception as e:
        return {"error": str(e)}


async def chroma_query_documents(
    collection_name: str,
    query_texts: list[str],
    n_results: int = 5,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
    include: list[str] = ["documents", "metadatas", "distances"],
) -> dict:
    """Query documents from a Chroma collection with advanced filtering and correct embedding dimension."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        embedding_dim = None
        if hasattr(collection, "metadata") and collection.metadata:
            embedding_dim = collection.metadata.get("embedding_dim")
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True
        )
        model_dim = model.get_sentence_embedding_dimension()
        if embedding_dim and embedding_dim != model_dim:
            return {
                "error": f"Model embedding dim {model_dim} != collection dim {embedding_dim}"
            }
        # Encode embedding cho query
        query_embeddings = model.encode(query_texts)
        if query_embeddings.shape[1] != model_dim:
            return {
                "error": f"Query embedding shape mismatch: {query_embeddings.shape[1]} != {model_dim}"
            }
        include_literal = cast(
            list[
                Literal[
                    "documents", "embeddings", "metadatas", "distances", "uris", "data"
                ]
            ],
            include,
        )
        result = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=n_results,
            where=where,
            where_document=where_document if where_document is not None else None,
            include=include_literal,
        )
        return dict(result)
    except Exception as e:
        return {"error": str(e)}


async def chroma_get_documents(
    collection_name: str,
    ids: Optional[list[str]] = None,
    where: Optional[dict] = None,
    where_document: Optional[dict] = None,
    include: list[str] = ["documents", "metadatas"],
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> dict:
    """Get documents from a Chroma collection with optional filtering."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        include_literal = cast(
            list[
                Literal[
                    "documents", "embeddings", "metadatas", "distances", "uris", "data"
                ]
            ],
            include,
        )
        result = collection.get(
            ids=ids if ids is not None else None,
            where=where,
            where_document=where_document if where_document is not None else None,
            include=include_literal,
            limit=limit,
            offset=offset,
        )
        return dict(result)
    except Exception as e:
        return {"error": str(e)}


async def chroma_update_documents(
    collection_name: str,
    ids: list[str],
    embeddings: Optional[list[list[float]]] = None,
    metadatas: Optional[list[dict]] = None,
    documents: Optional[list[str]] = None,
) -> dict:
    """Update documents in a Chroma collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        collection.update(
            ids=list(ids),
            embeddings=list(embeddings) if embeddings else None,
            metadatas=list(metadatas) if metadatas else None,
            documents=list(documents) if documents else None,
        )
        return {
            "result": f"Updated {len(ids)} documents in collection '{collection_name}'."
        }
    except Exception as e:
        return {"error": str(e)}


async def chroma_delete_documents(collection_name: str, ids: list[str]) -> dict:
    """Delete documents from a Chroma collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        collection.delete(ids=list(ids))
        return {
            "result": f"Deleted {len(ids)} documents from collection '{collection_name}'."
        }
    except Exception as e:
        return {"error": str(e)}


def validate_thought_data(input_data: Dict) -> Dict:
    """Validate thought data structure."""
    if not input_data.get("sessionId"):
        raise ValueError("Invalid sessionId: must be provided")
    if not input_data.get("thought") or not isinstance(input_data.get("thought"), str):
        raise ValueError("Invalid thought: must be a string")
    if not input_data.get("thoughtNumber") or not isinstance(
        input_data.get("thoughtNumber"), int
    ):
        raise ValueError("Invalid thoughtNumber: must be a number")
    if not input_data.get("totalThoughts") or not isinstance(
        input_data.get("totalThoughts"), int
    ):
        raise ValueError("Invalid totalThoughts: must be a number")
    if not isinstance(input_data.get("nextThoughtNeeded"), bool):
        raise ValueError("Invalid nextThoughtNeeded: must be a boolean")

    return {
        "sessionId": input_data.get("sessionId"),
        "thought": input_data.get("thought"),
        "thoughtNumber": input_data.get("thoughtNumber"),
        "totalThoughts": input_data.get("totalThoughts"),
        "nextThoughtNeeded": input_data.get("nextThoughtNeeded"),
        "isRevision": input_data.get("isRevision"),
        "revisesThought": input_data.get("revisesThought"),
        "branchFromThought": input_data.get("branchFromThought"),
        "branchId": input_data.get("branchId"),
        "needsMoreThoughts": input_data.get("needsMoreThoughts"),
    }


async def echo_echo(message: str) -> str:
    """Echo back the provided message with formatting."""
    return f"Echo: {message}"


async def echo_get_server_status() -> str:
    """Get the current server status."""
    return "ChromaDB MCP Server is running successfully!"


@app.call_tool()
async def handle_all_tools(name: str, arguments: dict) -> list[types.ContentBlock]:
    """Handle all tool calls in one unified handler."""

    # Handle server tools
    match name:
        case "chroma_get_server_status":
            status_message = await get_server_status()
            return [types.TextContent(type="text", text=status_message)]
        case "chroma_echo":
            message = arguments.get("message", "")
            echo_result = await echo(message)
            return [types.TextContent(type="text", text=echo_result)]
        case "chroma_list_collections":
            try:
                client = get_chroma_client()
                collection_names = await chroma_list_collections(
                    limit=arguments.get("limit"),
                    offset=arguments.get("offset"),
                )
                return [
                    types.TextContent(
                        type="text",
                        text=f"Collections: {json.dumps(collection_names, indent=2, ensure_ascii=False)}",
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_create_collection":
            try:
                collection_name = arguments.get("collection_name")
                embedding_function_name = arguments.get(
                    "embedding_function_name", "default"
                )
                metadata = arguments.get("metadata")

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_create_collection(
                    collection_name=collection_name,
                    embedding_function_name=embedding_function_name,
                    metadata=metadata,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_get_collection_info":
            try:
                collection_name = arguments.get("collection_name")
                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_get_collection_info(collection_name)
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_get_collection_count":
            try:
                collection_name = arguments.get("collection_name")
                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                count = await chroma_get_collection_count(collection_name)
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(count, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_modify_collection":
            try:
                collection_name = arguments.get("collection_name")
                new_name = arguments.get("new_name")
                new_metadata = arguments.get("new_metadata")

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_modify_collection(
                    collection_name=collection_name,
                    new_name=new_name,
                    new_metadata=new_metadata,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_delete_collection":
            try:
                collection_name = arguments.get("collection_name")
                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_delete_collection(collection_name)
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_add_documents":
            try:
                collection_name = arguments.get("collection_name")
                documents = arguments.get("documents", [])
                ids = arguments.get("ids", [])
                metadatas = arguments.get("metadatas")

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_add_documents(
                    collection_name=collection_name,
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_query_documents":
            try:
                collection_name = arguments.get("collection_name")
                query_texts = arguments.get("query_texts", [])
                n_results = arguments.get("n_results", 5)
                where = arguments.get("where")
                where_document = arguments.get("where_document")
                include = arguments.get(
                    "include", ["documents", "metadatas", "distances"]
                )

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_query_documents(
                    collection_name=collection_name,
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_get_documents":
            try:
                collection_name = arguments.get("collection_name")
                ids = arguments.get("ids")
                where = arguments.get("where")
                where_document = arguments.get("where_document")
                include = arguments.get("include", ["documents", "metadatas"])
                limit = arguments.get("limit")
                offset = arguments.get("offset")

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_get_documents(
                    collection_name=collection_name,
                    ids=ids,
                    where=where,
                    where_document=where_document,
                    include=include,
                    limit=limit,
                    offset=offset,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_update_documents":
            try:
                collection_name = arguments.get("collection_name")
                ids = arguments.get("ids", [])
                embeddings = arguments.get("embeddings")
                metadatas = arguments.get("metadatas")
                documents = arguments.get("documents")

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_update_documents(
                    collection_name=collection_name,
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_delete_documents":
            try:
                collection_name = arguments.get("collection_name")
                ids = arguments.get("ids", [])

                if not collection_name:
                    return [
                        types.TextContent(
                            type="text", text="Error: collection_name is required"
                        )
                    ]

                result = await chroma_delete_documents(
                    collection_name=collection_name,
                    ids=ids,
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case _:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    tools = []

    # Add server status tool
    tools.extend(
        [
            types.Tool(
                name="chroma_get_server_status",
                description="Get the current status of the MCP server.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="chroma_echo",
                description="Echo back the provided message.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back",
                        }
                    },
                    "required": ["message"],
                },
            ),
            types.Tool(
                name="chroma_list_collections",
                description="List all collection names in the Chroma database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Optional maximum number of collections to return",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Optional number of collections to skip",
                        },
                    },
                },
            ),
            types.Tool(
                name="chroma_create_collection",
                description="Create a new Chroma collection with configurable embedding function.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to create",
                        },
                        "embedding_function_name": {
                            "type": "string",
                            "description": "Name of the embedding function to use",
                            "default": "default",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata dict to add to the collection",
                        },
                    },
                    "required": ["collection_name"],
                },
            ),
            types.Tool(
                name="chroma_get_collection_info",
                description="Get information about a Chroma collection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to get info about",
                        }
                    },
                    "required": ["collection_name"],
                },
            ),
            types.Tool(
                name="chroma_get_collection_count",
                description="Get the number of documents in a Chroma collection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to count",
                        }
                    },
                    "required": ["collection_name"],
                },
            ),
            types.Tool(
                name="chroma_modify_collection",
                description="Modify a Chroma collection's name or metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to modify",
                        },
                        "new_name": {
                            "type": "string",
                            "description": "Optional new name for the collection",
                        },
                        "new_metadata": {
                            "type": "object",
                            "description": "Optional new metadata for the collection",
                        },
                    },
                    "required": ["collection_name"],
                },
            ),
            types.Tool(
                name="chroma_delete_collection",
                description="Delete a Chroma collection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to delete",
                        }
                    },
                    "required": ["collection_name"],
                },
            ),
            types.Tool(
                name="chroma_add_documents",
                description="Add documents to a Chroma collection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to add documents to",
                        },
                        "documents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of text documents to add",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of IDs for the documents (required)",
                        },
                        "metadatas": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Optional list of metadata dictionaries for each document",
                        },
                    },
                    "required": ["collection_name", "documents", "ids"],
                },
            ),
            types.Tool(
                name="chroma_query_documents",
                description="Query documents from a Chroma collection with advanced filtering.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to query",
                        },
                        "query_texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of query texts to search for",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return per query",
                            "default": 5,
                        },
                        "where": {
                            "type": "object",
                            "description": "Optional metadata filters using Chroma's query operators",
                        },
                        "where_document": {
                            "type": "object",
                            "description": "Optional document content filters",
                        },
                        "include": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of what to include in response",
                            "default": ["documents", "metadatas", "distances"],
                        },
                    },
                    "required": ["collection_name", "query_texts"],
                },
            ),
            types.Tool(
                name="chroma_get_documents",
                description="Get documents from a Chroma collection with optional filtering.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to get documents from",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of document IDs to retrieve",
                        },
                        "where": {
                            "type": "object",
                            "description": "Optional metadata filters using Chroma's query operators",
                        },
                        "where_document": {
                            "type": "object",
                            "description": "Optional document content filters",
                        },
                        "include": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of what to include in response",
                            "default": ["documents", "metadatas"],
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional maximum number of documents to return",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Optional number of documents to skip before returning results",
                        },
                    },
                    "required": ["collection_name"],
                },
            ),
            types.Tool(
                name="chroma_update_documents",
                description="Update documents in a Chroma collection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to update documents in",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs to update (required)",
                        },
                        "embeddings": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Optional list of new embeddings for the documents",
                        },
                        "metadatas": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Optional list of new metadata dictionaries for the documents",
                        },
                        "documents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of new text documents",
                        },
                    },
                    "required": ["collection_name", "ids"],
                },
            ),
            types.Tool(
                name="chroma_delete_documents",
                description="Delete documents from a Chroma collection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to delete documents from",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs to delete",
                        },
                    },
                    "required": ["collection_name", "ids"],
                },
            ),
        ]
    )

    return tools


def main() -> None:
    """Main function to run the ChromaDB MCP server"""

    # Tạo SentenceTransformer thủ công
    model = load_embedding_function()

    # Khởi tạo ChromaDB client với API mới
    client = get_chroma_client()

    # Tạo hoặc lấy collection
    collection_name = os.getenv("CHROMADB_COLLECTION_NAME", "nomic_embeddings")
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    # Thêm dữ liệu mẫu với embedding thủ công
    sentences = ["Hello!", "¡Hola!", "Xin chào!", "こんにちは！"]
    embeddings = model(sentences)

    ids = [str(uuid.uuid4()) for _ in sentences]
    collection.add(documents=sentences, embeddings=embeddings, ids=ids)
    logger.info("Đã thêm dữ liệu với embedding thủ công vào ChromaDB.")

    data = client.get_collection(collection_name)
    logger.info("Dữ liệu trong collection:", data)

    try:
        import anyio
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)
    except Exception as e:
        logger.info(f"Error running server: {e}")
    finally:
        logger.info("Shutting down server...")
        os._exit(0)


if __name__ == "__main__":
    main()
