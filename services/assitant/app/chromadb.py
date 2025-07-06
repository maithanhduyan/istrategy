# -*- coding: utf-8 -*-
# app/chroma.py

import chromadb
from typing import List, Dict, Optional
from chromadb.config import Settings
from app.logger import get_logger
from app.config import CHROMA_DB_CONFIG

logger = get_logger(__name__)

# Global variables
_chroma_client = None

def get_chroma_client():
    """Get the ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        try:
            # Try to create persistent client first
            data_dir = CHROMA_DB_CONFIG.get("persist_directory", "./chroma_db")
            _chroma_client = chromadb.PersistentClient(path=data_dir)
            logger.info(f"ChromaDB client initialized with data directory: {data_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            # Fallback to ephemeral client
            _chroma_client = chromadb.EphemeralClient()
            logger.info("Falling back to ephemeral ChromaDB client.")
            raise

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
        logger.error(f"Error listing collections: {e}")
        raise


async def chroma_create_collection(
    collection_name: str,
    embedding_function_name: str = "default",
    metadata: Optional[dict] = None,
) -> str:
    """Create a new Chroma collection with configurable HNSW parameters.

    Args:
        collection_name: Name of the collection to create
        embedding_function_name: Name of the embedding function to use. Options: 'default', 'cohere', 'openai', 'jina', 'voyageai', 'ollama', 'roboflow'
        metadata: Optional metadata dict to add to the collection
    """
    client = get_chroma_client()
    try:
        embedding_dimensions =   {
            "default": 1536,
            "cohere": 1024,
            "openai": 1536,
            "jina": 768,
            "voyageai": 512,
            "ollama": 4096,
            "roboflow": 512,
        }.get(embedding_function_name, 1536)
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function_name,
            metadata=metadata or {},
        )
        return {
            "result": f"Collection '{collection}' created successfully.",
            "embedding_dim": embedding_dimensions,
        }
    except Exception as e:
        raise Exception(
            f"Failed to create collection '{collection_name}': {str(e)}"
        ) from e

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
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        # Ensure metadatas is a list of dicts, default to empty dict if None
        metadatas = metadatas or [{}] * len(documents)
        
        # Add documents with their IDs and metadata
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )
        return {"result": f"Added {len(documents)} documents to collection '{collection_name}'."}
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
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        # Perform the query
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )
        
        return {
            "result": f"Queried {len(query_texts)} texts from collection '{collection_name}'.",
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
            "distances": results.get("distances", []),
        }
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
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        # Get documents with optional filtering
        results = collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset,
        )
        
        return {
            "result": f"Retrieved {len(results.get('documents', []))} documents from collection '{collection_name}'.",
            "documents": results.get("documents", []),
            "metadatas": results.get("metadatas", []),
        }
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
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        # Prepare the update data
        update_data = {}
        if embeddings is not None:
            update_data["embeddings"] = embeddings
        if metadatas is not None:
            update_data["metadatas"] = metadatas
        if documents is not None:
            update_data["documents"] = documents
        
        # Update the documents
        collection.update(
            ids=ids,
            **update_data,
        )
        
        return {"result": f"Updated {len(ids)} documents in collection '{collection_name}'."}
    except Exception as e:
        return {"error": str(e)}

async def chroma_delete_documents(collection_name: str, ids: list[str]) -> dict:
    """Delete documents from a Chroma collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        # Delete the documents by their IDs
        collection.delete(ids=ids)
        
        return {"result": f"Deleted {len(ids)} documents from collection '{collection_name}'."}
    except Exception as e:
        return {"error": str(e)}
    
async def chroma_get_collection_stats(collection_name: str) -> dict:
    """Get statistics about a Chroma collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        stats = {
            "name": collection.name,
            "count": collection.count(),
            "metadata": getattr(collection, "metadata", {}),
        }
        
        return {"result": f"Statistics for collection '{collection_name}' retrieved successfully.", "stats": stats}
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



