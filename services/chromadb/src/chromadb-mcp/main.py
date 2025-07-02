import contextlib
import logging
import json
import mcp.types as types
from collections.abc import AsyncIterator
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from typing import Dict, List, TypedDict, Union
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send
import argparse
import os
import chromadb


from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.api import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    CohereEmbeddingFunction,
    OpenAIEmbeddingFunction,
    JinaEmbeddingFunction,
    VoyageAIEmbeddingFunction,
    RoboflowEmbeddingFunction,
)


mcp_known_embedding_functions: Dict[str, EmbeddingFunction] = {
    "default": DefaultEmbeddingFunction,
    "cohere": CohereEmbeddingFunction,
    "openai": OpenAIEmbeddingFunction,
    "jina": JinaEmbeddingFunction,
    "voyageai": VoyageAIEmbeddingFunction,
    "roboflow": RoboflowEmbeddingFunction,
}


# Create MCP server using low-level Server for StreamableHTTP compatibility
app = Server("chromadb-mcp")

# Global variables
_chroma_client = None


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="FastMCP server for Chroma DB")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("CHROMA_DATA_DIR"),
        help="Directory for persistent client data (only used with persistent client)",
    )
    return parser


def get_chroma_client(args=None):
    """Get or create the global Chroma client instance."""
    global _chroma_client

    if _chroma_client is None:
        try:
            # Try to create persistent client first
            data_dir = os.getenv("CHROMA_DATA_DIR", "./chromadb")
            _chroma_client = chromadb.PersistentClient(path=data_dir)
        except Exception:
            # Fallback to ephemeral client
            _chroma_client = chromadb.EphemeralClient()

    return _chroma_client


async def chroma_list_collections(
    limit: int | None = None, offset: int | None = None
) -> List[str]:
    """List all collection names in the Chroma database with pagination support.

    Args:
        limit: Optional maximum number of collections to return
        offset: Optional number of collections to skip before returning results

    Returns:
        List of collection names or ["__NO_COLLECTIONS_FOUND__"] if database is empty
    """
    client = get_chroma_client()
    try:
        colls = client.list_collections(limit=limit, offset=offset)
        # Safe handling: If colls is None or empty, return a special marker
        if not colls:
            return ["__NO_COLLECTIONS_FOUND__"]
        # Otherwise iterate to get collection names
        return [coll.name for coll in colls]

    except Exception as e:
        raise Exception(f"Failed to list collections: {str(e)}") from e


async def chroma_create_collection(
    collection_name: str,
    embedding_function_name: str = "default",
    metadata: Dict | None = None,
) -> str:
    """Create a new Chroma collection with configurable HNSW parameters.

    Args:
        collection_name: Name of the collection to create
        embedding_function_name: Name of the embedding function to use. Options: 'default', 'cohere', 'openai', 'jina', 'voyageai', 'ollama', 'roboflow'
        metadata: Optional metadata dict to add to the collection
    """
    client = get_chroma_client()

    embedding_function = mcp_known_embedding_functions[embedding_function_name]

    configuration = CreateCollectionConfiguration(
        embedding_function=embedding_function()
    )

    try:
        client.create_collection(
            name=collection_name, configuration=configuration, metadata=metadata
        )
        config_msg = f" with configuration: {configuration}"
        return f"Successfully created collection {collection_name}{config_msg}"
    except Exception as e:
        raise Exception(
            f"Failed to create collection '{collection_name}': {str(e)}"
        ) from e


async def chroma_get_collection_info(collection_name: str) -> Dict:
    """Get information about a Chroma collection.

    Args:
        collection_name: Name of the collection to get info about
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)

        # Get collection count
        count = collection.count()

        # Peek at a few documents
        peek_results = collection.peek(limit=3)
        
        # Convert numpy arrays to lists for JSON serialization
        safe_peek_results = {}
        for key, value in peek_results.items():
            if hasattr(value, 'tolist'):  # numpy array
                safe_peek_results[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
            else:
                safe_peek_results[key] = value

        return {
            "name": collection_name,
            "count": count,
            "sample_documents": safe_peek_results,
        }
    except Exception as e:
        raise Exception(
            f"Failed to get collection info for '{collection_name}': {str(e)}"
        ) from e


async def chroma_get_collection_count(collection_name: str) -> int:
    """Get the number of documents in a Chroma collection.

    Args:
        collection_name: Name of the collection to count
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.count()
    except Exception as e:
        raise Exception(
            f"Failed to get collection count for '{collection_name}': {str(e)}"
        ) from e


async def chroma_modify_collection(
    collection_name: str,
    new_name: str | None = None,
    new_metadata: Dict | None = None,
) -> str:
    """Modify a Chroma collection's name or metadata.

    Args:
        collection_name: Name of the collection to modify
        new_name: Optional new name for the collection
        new_metadata: Optional new metadata for the collection
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        collection.modify(name=new_name, metadata=new_metadata)

        modified_aspects = []
        if new_name:
            modified_aspects.append("name")
        if new_metadata:
            modified_aspects.append("metadata")

        return f"Successfully modified collection {collection_name}: updated {' and '.join(modified_aspects)}"
    except Exception as e:
        raise Exception(
            f"Failed to modify collection '{collection_name}': {str(e)}"
        ) from e


async def chroma_delete_collection(collection_name: str) -> str:
    """Delete a Chroma collection.

    Args:
        collection_name: Name of the collection to delete
    """
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        return f"Successfully deleted collection {collection_name}"
    except Exception as e:
        raise Exception(
            f"Failed to delete collection '{collection_name}': {str(e)}"
        ) from e


async def chroma_add_documents(
    collection_name: str,
    documents: List[str],
    ids: List[str],
    metadatas: List[Dict] | None = None,
) -> str:
    """Add documents to a Chroma collection.

    Args:
        collection_name: Name of the collection to add documents to
        documents: List of text documents to add
        ids: List of IDs for the documents (required)
        metadatas: Optional list of metadata dictionaries for each document
    """
    if not documents:
        raise ValueError("The 'documents' list cannot be empty.")

    if not ids:
        raise ValueError("The 'ids' list is required and cannot be empty.")

    # Check if there are empty strings in the ids list
    if any(not id.strip() for id in ids):
        raise ValueError("IDs cannot be empty strings.")

    if len(ids) != len(documents):
        raise ValueError(
            f"Number of ids ({len(ids)}) must match number of documents ({len(documents)})."
        )

    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(collection_name)

        # Check for duplicate IDs
        existing_ids = collection.get(include=[])["ids"]
        duplicate_ids = [id for id in ids if id in existing_ids]

        if duplicate_ids:
            raise ValueError(
                f"The following IDs already exist in collection '{collection_name}': {duplicate_ids}. "
                f"Use 'chroma_update_documents' to update existing documents."
            )

        result = collection.add(documents=documents, metadatas=metadatas, ids=ids)

        # Check the return value
        if result and isinstance(result, dict):
            # If the return value is a dictionary, it may contain success information
            if "success" in result and not result["success"]:
                raise Exception(
                    f"Failed to add documents: {result.get('error', 'Unknown error')}"
                )

            # If the return value contains the actual number added
            if "count" in result:
                return f"Successfully added {result['count']} documents to collection {collection_name}"

        # Default return
        return f"Successfully added {len(documents)} documents to collection {collection_name}, result is {result}"
    except Exception as e:
        raise Exception(
            f"Failed to add documents to collection '{collection_name}': {str(e)}"
        ) from e


async def chroma_query_documents(
    collection_name: str,
    query_texts: List[str],
    n_results: int = 5,
    where: Dict | None = None,
    where_document: Dict | None = None,
    include: List[str] = ["documents", "metadatas", "distances"],
) -> Dict:
    """Query documents from a Chroma collection with advanced filtering.

    Args:
        collection_name: Name of the collection to query
        query_texts: List of query texts to search for
        n_results: Number of results to return per query
        where: Optional metadata filters using Chroma's query operators
               Examples:
               - Simple equality: {"metadata_field": "value"}
               - Comparison: {"metadata_field": {"$gt": 5}}
               - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: Optional document content filters
        include: List of what to include in response. By default, this will include documents, metadatas, and distances.
    """
    if not query_texts:
        raise ValueError("The 'query_texts' list cannot be empty.")

    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )
    except Exception as e:
        raise Exception(
            f"Failed to query documents from collection '{collection_name}': {str(e)}"
        ) from e


async def chroma_get_documents(
    collection_name: str,
    ids: List[str] | None = None,
    where: Dict | None = None,
    where_document: Dict | None = None,
    include: List[str] = ["documents", "metadatas"],
    limit: int | None = None,
    offset: int | None = None,
) -> Dict:
    """Get documents from a Chroma collection with optional filtering.

    Args:
        collection_name: Name of the collection to get documents from
        ids: Optional list of document IDs to retrieve
        where: Optional metadata filters using Chroma's query operators
               Examples:
               - Simple equality: {"metadata_field": "value"}
               - Comparison: {"metadata_field": {"$gt": 5}}
               - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: Optional document content filters
        include: List of what to include in response. By default, this will include documents, and metadatas.
        limit: Optional maximum number of documents to return
        offset: Optional number of documents to skip before returning results

    Returns:
        Dictionary containing the matching documents, their IDs, and requested includes
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise Exception(
            f"Failed to get documents from collection '{collection_name}': {str(e)}"
        ) from e


async def chroma_update_documents(
    collection_name: str,
    ids: List[str],
    embeddings: List[List[float]] | None = None,
    metadatas: List[Dict] | None = None,
    documents: List[str] | None = None,
) -> str:
    """Update documents in a Chroma collection.

    Args:
        collection_name: Name of the collection to update documents in
        ids: List of document IDs to update (required)
        embeddings: Optional list of new embeddings for the documents.
                    Must match length of ids if provided.
        metadatas: Optional list of new metadata dictionaries for the documents.
                   Must match length of ids if provided.
        documents: Optional list of new text documents.
                   Must match length of ids if provided.

    Returns:
        A confirmation message indicating the number of documents updated.

    Raises:
        ValueError: If 'ids' is empty or if none of 'embeddings', 'metadatas',
                    or 'documents' are provided, or if the length of provided
                    update lists does not match the length of 'ids'.
        Exception: If the collection does not exist or if the update operation fails.
    """
    if not ids:
        raise ValueError("The 'ids' list cannot be empty.")

    if embeddings is None and metadatas is None and documents is None:
        raise ValueError(
            "At least one of 'embeddings', 'metadatas', or 'documents' "
            "must be provided for update."
        )

    # Ensure provided lists match the length of ids if they are not None
    if embeddings is not None and len(embeddings) != len(ids):
        raise ValueError("Length of 'embeddings' list must match length of 'ids' list.")
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("Length of 'metadatas' list must match length of 'ids' list.")
    if documents is not None and len(documents) != len(ids):
        raise ValueError("Length of 'documents' list must match length of 'ids' list.")

    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        raise Exception(
            f"Failed to get collection '{collection_name}': {str(e)}"
        ) from e

    # Prepare arguments for update, excluding None values at the top level
    update_args = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": documents,
    }
    kwargs = {k: v for k, v in update_args.items() if v is not None}

    try:
        collection.update(**kwargs)
        return (
            f"Successfully processed update request for {len(ids)} documents in "
            f"collection '{collection_name}'. Note: Non-existent IDs are ignored by ChromaDB."
        )
    except Exception as e:
        raise Exception(
            f"Failed to update documents in collection '{collection_name}': {str(e)}"
        ) from e

async def chroma_delete_documents(
    collection_name: str,
    ids: List[str]
) -> str:
    """Delete documents from a Chroma collection.

    Args:
        collection_name: Name of the collection to delete documents from
        ids: List of document IDs to delete

    Returns:
        A confirmation message indicating the number of documents deleted.

    Raises:
        ValueError: If 'ids' is empty
        Exception: If the collection does not exist or if the delete operation fails.
    """
    if not ids:
        raise ValueError("The 'ids' list cannot be empty.")

    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        raise Exception(
            f"Failed to get collection '{collection_name}': {str(e)}"
        ) from e

    try:
        collection.delete(ids=ids)
        return (
            f"Successfully deleted {len(ids)} documents from "
            f"collection '{collection_name}'. Note: Non-existent IDs are ignored by ChromaDB."
        )
    except Exception as e:
        raise Exception(
            f"Failed to delete documents from collection '{collection_name}': {str(e)}"
        ) from e


def validate_thought_data(input_data: Dict) -> Dict:
    """Validate thought data structure."""
    if not input_data.get("sessionId"):
        raise ValueError("Invalid sessionId: must be provided")
    if not input_data.get("thought") or not isinstance(input_data.get("thought"), str):
        raise ValueError("Invalid thought: must be a string")
    if not input_data.get("thoughtNumber") or not isinstance(input_data.get("thoughtNumber"), int):
            raise ValueError("Invalid thoughtNumber: must be a number")
    if not input_data.get("totalThoughts") or not isinstance(input_data.get("totalThoughts"), int):
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


def echo(message: str) -> str:
    """Echo back the provided message with formatting."""
    return f"Echo: {message}"


def get_server_status() -> str:
    """Get the current server status."""
    return "ChromaDB MCP Server is running successfully!"


# Unified tool handler
@app.call_tool()
async def handle_all_tools(name: str, arguments: dict) -> list[types.ContentBlock]:
    """Handle all tool calls in one unified handler."""

    # Handle server tools
    match name:
        case "get_server_status":
            status_message = get_server_status()
            return [types.TextContent(type="text", text=status_message)]
        case "echo":
            message = arguments.get("message", "")
            echo_result = echo(message)
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
                        text=f"Collections: {json.dumps(collection_names, indent=2)}",
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
                return [types.TextContent(type="text", text=result)]
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
                        text=f"Collection info: {json.dumps(result, indent=2)}",
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
                        text=f"Collection '{collection_name}' contains {count} documents",
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
                return [types.TextContent(type="text", text=result)]
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
                return [types.TextContent(type="text", text=result)]
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
                return [types.TextContent(type="text", text=result)]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        case "chroma_query_documents":
            try:
                collection_name = arguments.get("collection_name")
                query_texts = arguments.get("query_texts", [])
                n_results = arguments.get("n_results", 5)
                where = arguments.get("where")
                where_document = arguments.get("where_document")
                include = arguments.get("include", ["documents", "metadatas", "distances"])
                
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
                        text=f"Query results: {json.dumps(result, indent=2)}",
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
                        text=f"Documents: {json.dumps(result, indent=2)}",
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
                return [types.TextContent(type="text", text=result)]
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
                return [types.TextContent(type="text", text=result)]
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
                name="get_server_status",
                description="Get the current status of the MCP server.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="echo",
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
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
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
            logger.info(
                "ChromaDB MCP server started with StreamableHTTP session manager!"
            )
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
