"""Qdrant tools for MCP server."""

from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging
import uuid

# Setup logging
logger = logging.getLogger(__name__)

# Global Qdrant client instance
_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(host="localhost", port=6333)
            logger.info("Connected to Qdrant at localhost:6333")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    return _qdrant_client

def register_qdrant_tools(mcp_server: FastMCP):
    """Register Qdrant-related tools with MCP server."""
    
    @mcp_server.tool()
    async def qdrant_status() -> Dict:
        """Get Qdrant connection status."""
        try:
            client = get_qdrant_client()
            collections = client.get_collections()
            return {
                "status": "connected",
                "host": "localhost", 
                "port": 6333,
                "collections": len(collections.collections),
                "version": "1.0.0-real"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "host": "localhost",
                "port": 6333
            }
    
    @mcp_server.tool()
    async def list_collections() -> List[str]:
        """List all collections in Qdrant."""
        try:
            client = get_qdrant_client()
            collections = client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return [f"Error: {str(e)}"]
    
    @mcp_server.tool()
    async def create_collection(name: str, vector_size: int = 384) -> Dict:
        """Create a new collection in Qdrant."""
        try:
            client = get_qdrant_client()
            
            # Check if collection already exists
            try:
                client.get_collection(name)
                return {
                    "collection_name": name,
                    "vector_size": vector_size,
                    "status": "already_exists",
                    "message": f"Collection '{name}' already exists"
                }
            except:
                # Collection doesn't exist, create it
                pass
            
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            
            return {
                "collection_name": name,
                "vector_size": vector_size,
                "status": "created",
                "message": f"Collection '{name}' created successfully with vector size {vector_size}"
            }
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return {
                "collection_name": name,
                "status": "error",
                "error": str(e)
            }
    
    @mcp_server.tool()
    async def search_vectors(
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10
    ) -> Dict:
        """Search for similar vectors in Qdrant collection."""
        try:
            client = get_qdrant_client()
            
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload or {}
                })
            
            return {
                "collection": collection_name,
                "query_vector_size": len(query_vector),
                "limit": limit,
                "results": results,
                "found": len(results)
            }
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return {
                "collection": collection_name,
                "error": str(e),
                "results": []
            }
    
    @mcp_server.tool()
    async def insert_vectors(
        collection_name: str, 
        vectors: List[List[float]], 
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> Dict:
        """Insert vectors into Qdrant collection."""
        try:
            client = get_qdrant_client()
            
            if payloads is None:
                payloads = [{"id": i} for i in range(len(vectors))]
            
            # Create points with random UUIDs
            points = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                ))
            
            operation_info = client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            return {
                "collection": collection_name,
                "inserted_count": len(vectors),
                "vector_size": len(vectors[0]) if vectors else 0,
                "payloads_count": len(payloads),
                "status": "success",
                "operation_id": operation_info.operation_id if hasattr(operation_info, 'operation_id') else None
            }
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return {
                "collection": collection_name,
                "error": str(e),
                "status": "error"
            }
    
    @mcp_server.tool()
    async def get_collection_info(collection_name: str) -> Dict:
        """Get information about a Qdrant collection."""
        try:
            client = get_qdrant_client()
            
            collection_info = client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count or 0,
                "points_count": collection_info.points_count or 0,
                "segments_count": collection_info.segments_count or 0,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else 0,
                    "distance": str(collection_info.config.params.vectors.distance) if hasattr(collection_info.config.params, 'vectors') else "Unknown"
                },
                "status": str(collection_info.status)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": collection_name,
                "error": str(e),
                "status": "error"
            }


# Alias để dễ import
qdrant_tools = register_qdrant_tools
