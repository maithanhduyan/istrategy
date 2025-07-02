"""Text embedding pipeline."""

from typing import List, Dict, Any, Optional
import logging
from ..models.nomic import NomicEmbeddingModel
from ..schemas import NomicTextInput, EmbeddingResponse, EmbeddingBatchInput, EmbeddingBatchResponse

logger = logging.getLogger(__name__)


class TextEmbeddingPipeline:
    """Pipeline for text embedding processing."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self._default_model = "nomic"
    
    async def get_model(self, model_name: str = "nomic") -> Any:
        """Get or create embedding model."""
        if model_name not in self.models:
            if model_name == "nomic":
                self.models[model_name] = NomicEmbeddingModel()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        model = self.models[model_name]
        await model.ensure_loaded()
        return model
    
    async def create_embedding(self, input_data: NomicTextInput, model_name: str = "nomic") -> EmbeddingResponse:
        """Create single text embedding."""
        try:
            model = await self.get_model(model_name)
            
            # Generate embedding
            embeddings = await model.embed_text(
                text=input_data.text,
                prompt_name=input_data.prompt_name
            )
            
            logger.info(f"Generated {model_name} embedding for text: {input_data.text[:50]}... with prompt: {input_data.prompt_name}")
            
            return EmbeddingResponse(
                text=input_data.text,
                embeddings=embeddings,
                model=model.model_name,
                dimensions=len(embeddings),
                prompt_name=input_data.prompt_name
            )
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    async def create_batch_embeddings(self, input_data: EmbeddingBatchInput, model_name: str = "nomic") -> EmbeddingBatchResponse:
        """Create batch text embeddings."""
        try:
            model = await self.get_model(model_name)
            
            # Generate embeddings
            embeddings = await model.embed_batch(
                texts=input_data.texts,
                prompt_name=input_data.prompt_name
            )
            
            logger.info(f"Generated {model_name} batch embeddings for {len(input_data.texts)} texts")
            
            return EmbeddingBatchResponse(
                embeddings=embeddings,
                model=model.model_name,
                dimensions=len(embeddings[0]) if embeddings else 0,
                count=len(embeddings)
            )
            
        except Exception as e:
            logger.error(f"Failed to create batch embeddings: {e}")
            raise
    
    async def get_model_info(self, model_name: str = "nomic") -> Dict[str, Any]:
        """Get model information."""
        try:
            model = await self.get_model(model_name)
            return model.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def list_available_models(self) -> List[str]:
        """List available embedding models."""
        return ["nomic"]


# Global pipeline instance
_pipeline = None

async def get_embedding_pipeline() -> TextEmbeddingPipeline:
    """Get global embedding pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TextEmbeddingPipeline()
    return _pipeline
