"""Embedding tools for MCP server."""

from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Any
import logging
from ..embedding.pipeline.text_processor import get_embedding_pipeline
from ..embedding.schemas import NomicTextInput, EmbeddingBatchInput

logger = logging.getLogger(__name__)


def register_embedding_tools(mcp_server: FastMCP):
    """Register embedding-related tools with MCP server."""
    
    @mcp_server.tool()
    async def create_embedding(
        text: str, 
        prompt_name: str = "search_query",
        model: str = "nomic"
    ) -> Dict:
        """Create text embedding using specified model."""
        try:
            pipeline = await get_embedding_pipeline()
            input_data = NomicTextInput(text=text, prompt_name=prompt_name)
            
            result = await pipeline.create_embedding(input_data, model_name=model)
            
            return {
                "text": result.text,
                "embeddings": result.embeddings,
                "model": result.model,
                "dimensions": result.dimensions,
                "prompt_name": result.prompt_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp_server.tool()
    async def create_batch_embeddings(
        texts: List[str],
        prompt_name: str = "search_document", 
        model: str = "nomic"
    ) -> Dict:
        """Create batch text embeddings."""
        try:
            pipeline = await get_embedding_pipeline()
            input_data = EmbeddingBatchInput(texts=texts, prompt_name=prompt_name)
            
            result = await pipeline.create_batch_embeddings(input_data, model_name=model)
            
            return {
                "embeddings": result.embeddings,
                "model": result.model,
                "dimensions": result.dimensions,
                "count": result.count,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to create batch embeddings: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp_server.tool()
    async def get_embedding_model_info(model: str = "nomic") -> Dict:
        """Get information about embedding model."""
        try:
            pipeline = await get_embedding_pipeline()
            info = await pipeline.get_model_info(model_name=model)
            
            return {
                "model_info": info,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp_server.tool()
    async def list_embedding_models() -> Dict:
        """List available embedding models."""
        try:
            pipeline = await get_embedding_pipeline()
            models = pipeline.list_available_models()
            
            return {
                "available_models": models,
                "default_model": "nomic",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    @mcp_server.tool()
    async def embed_and_store_in_qdrant(
        text: str,
        collection_name: str,
        payload: Dict[str, Any] = None,
        prompt_name: str = "search_document",
        model: str = "nomic"
    ) -> Dict:
        """Create embedding and store directly in Qdrant collection."""
        try:
            # Import Qdrant client
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct
            import uuid
            
            # Create embedding
            pipeline = await get_embedding_pipeline()
            input_data = NomicTextInput(text=text, prompt_name=prompt_name)
            embedding_result = await pipeline.create_embedding(input_data, model_name=model)
            
            # Store in Qdrant
            client = QdrantClient(host="localhost", port=6333)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding_result.embeddings,
                payload=payload or {"text": text, "model": embedding_result.model}
            )
            
            operation_info = client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            return {
                "text": text,
                "collection": collection_name,
                "embedding_dimensions": embedding_result.dimensions,
                "model": embedding_result.model,
                "point_id": point.id,
                "operation_id": operation_info.operation_id if hasattr(operation_info, 'operation_id') else None,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to embed and store in Qdrant: {e}")
            return {
                "error": str(e),
                "status": "error"
            }


# Alias để dễ import
embedding_tools = register_embedding_tools
