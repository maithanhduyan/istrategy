"""Base embedding model interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the embedding model."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str, **kwargs) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimension size of embeddings."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    async def ensure_loaded(self) -> None:
        """Ensure model is loaded before use."""
        if not self._is_loaded:
            await self.load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "dimensions": self.get_dimensions() if self._is_loaded else None,
            "loaded": self._is_loaded
        }
