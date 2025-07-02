"""Nomic AI embedding model implementation."""

from typing import List, Dict, Any
import logging
from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class NomicEmbeddingModel(BaseEmbeddingModel):
    """Nomic AI text embedding model implementation.
    
    Valid prompt names for nomic-ai/nomic-embed-text-v2-moe:
        - 'query'
        - 'document'
        - 'passage'
        - 'Classification'
        - 'MultilabelClassification'
        - 'Clustering'
        - 'PairClassification'
        - 'STS'
        - 'Summarization'
        - 'Speed'
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v2-moe"):
        super().__init__(model_name)
        self.valid_prompt_names = [
            "query",
            "document",
            "passage",
            "Classification",
            "MultilabelClassification",
            "Clustering",
            "PairClassification",
            "STS",
            "Summarization",
            "Speed"
        ]
    
    async def load_model(self) -> None:
        """Load the Nomic embedding model."""
        try:
            # Import here to avoid loading at startup
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading Nomic model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name, 
                trust_remote_code=True
            )
            self._is_loaded = True
            logger.info(f"Successfully loaded Nomic model with {self.get_dimensions()} dimensions")
            
        except Exception as e:
            logger.error(f"Failed to load Nomic model: {e}")
            raise RuntimeError(f"Could not load Nomic model: {e}")
    
    async def embed_text(self, text: str, prompt_name: str = "query") -> List[float]:
        """Generate embedding for a single text."""
        await self.ensure_loaded()
        
        if prompt_name not in self.valid_prompt_names:
            logger.warning(f"Invalid prompt_name '{prompt_name}', using 'query'")
            prompt_name = "query"
        
        try:
            embedding = self.model.encode(text, prompt_name=prompt_name)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    async def embed_batch(self, texts: List[str], prompt_name: str = "document") -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        await self.ensure_loaded()
        
        if prompt_name not in self.valid_prompt_names:
            logger.warning(f"Invalid prompt_name '{prompt_name}', using 'document'")
            prompt_name = "document"
        
        try:
            embeddings = self.model.encode(texts, prompt_name=prompt_name)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")
    
    def get_dimensions(self) -> int:
        """Get the dimension size of Nomic embeddings."""
        if not self._is_loaded:
            # Nomic v2 MoE has 768 dimensions
            return 768
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Nomic model information."""
        info = super().get_model_info()
        info.update({
            "type": "nomic",
            "valid_prompt_names": self.valid_prompt_names,
            "default_prompt_single": "search_query",
            "default_prompt_batch": "search_document"
        })
        return info
