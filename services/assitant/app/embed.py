# -*- coding: utf-8 -*-
# app/embed.py
from fastapi import HTTPException
from pydantic import BaseModel
from app.logger import get_logger


EMBED_TABLE = "test_embeddings"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"



logger = get_logger(__name__)

class TextInput(BaseModel):
    text: str

def get_model():
    """Get the sentence-transformers model."""
    if not hasattr(get_model, "_model"):
        logger.info(f"Model '{EMBED_MODEL_NAME}' is loading...")
        from sentence_transformers import SentenceTransformer
        get_model._model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info(f"Model '{EMBED_MODEL_NAME}' loaded successfully.")
    else:
        logger.info(f"Using cached model '{EMBED_MODEL_NAME}'.")
    return get_model._model

async def create_embedding(input_data: TextInput):
    """Create text embeddings using sentence-transformers."""
    try:
        model = get_model()
        embeddings = model.encode(input_data.text).tolist()
        logger.info(f"Generated embeddings for text: {input_data.text[:50]}...")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Error generating embeddings")