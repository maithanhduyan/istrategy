"""Data schemas for embedding pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field


class NomicTextInput(BaseModel):
    """Input schema for Nomic text embedding."""
    text: str = Field(..., description="Text to embed")
    prompt_name: str = Field(
        default="search_query",
        description="Prompt name for Nomic model (search_query, search_document, classification, clustering)"
    )


class EmbeddingResponse(BaseModel):
    """Response schema for embedding generation."""
    text: str = Field(..., description="Original input text")
    embeddings: List[float] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model name used for embedding")
    dimensions: int = Field(..., description="Number of dimensions in embedding")
    prompt_name: Optional[str] = Field(None, description="Prompt name used (for Nomic)")


class EmbeddingBatchInput(BaseModel):
    """Input schema for batch embedding."""
    texts: List[str] = Field(..., description="List of texts to embed")
    prompt_name: str = Field(
        default="search_document",
        description="Prompt name for all texts"
    )


class EmbeddingBatchResponse(BaseModel):
    """Response schema for batch embedding."""
    embeddings: List[List[float]] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model name used")
    dimensions: int = Field(..., description="Number of dimensions")
    count: int = Field(..., description="Number of embeddings generated")
