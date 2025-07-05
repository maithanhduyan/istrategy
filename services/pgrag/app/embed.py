from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from config import DB_CONFIG
from sentence_transformers import SentenceTransformer
import asyncpg
from logger import get_logger
from db import get_pool as db_get_pool, execute_query, fetch_one, fetch_all

EMBED_TABLE = "test_embeddings"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Không khởi tạo model tại module level để tránh reload issues
# MODEL = SentenceTransformer(MODEL_NAME)

logger = get_logger(__name__)


class TextInput(BaseModel):
    text: str

def get_model():
    """Get the sentence-transformers model."""
    if not hasattr(get_model, "_model"):
        get_model._model = SentenceTransformer(MODEL_NAME)
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

async def save_embedding(input_data: TextInput):
    """Save text embeddings to the database."""
    try:
        logger.info(f"Starting save_embedding for text: {input_data.text[:50]}...")
        model = get_model()
        embeddings = model.encode(input_data.text).tolist()
        logger.info(f"Generated {len(embeddings)} dimensional embedding")
        
        # Convert embedding list to string format for pgvector
        embedding_str = str(embeddings)
        
        sql = f"INSERT INTO {EMBED_TABLE} (text, embedding) VALUES ($1, $2) RETURNING id;"
        logger.info(f"Executing SQL: {sql}")
        result = await fetch_one(sql, [input_data.text, embedding_str])
        logger.info(f"Database result: {result}")
        return result["id"]
    except Exception as e:
        logger.error(f"Error saving embeddings to database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving embeddings to database: {str(e)}")

async def get_embedding(row_id: int):
    try:
        sql = f"SELECT id, text, embedding FROM {EMBED_TABLE} WHERE id = $1;"
        row = await fetch_one(sql, [row_id])
        if row:
            return {"id": row["id"], "text": row["text"], "embedding": row["embedding"]}
        raise HTTPException(status_code=404, detail="Embedding not found")
    except Exception as e:
        logger.error(f"Error fetching embedding: {e}")
        raise HTTPException(status_code=500, detail="Error fetching embedding")

async def search_embeddings(query: str):
    try:
        sql = f"SELECT id, text FROM {EMBED_TABLE} WHERE text ILIKE $1;"
        rows = await fetch_all(sql, [f"%{query}%"])
        return [{"id": row["id"], "text": row["text"]} for row in rows]
    except Exception as e:
        logger.error(f"Error searching embeddings: {e}")
        raise HTTPException(status_code=500, detail="Error searching embeddings")

async def update_embedding(row_id: int, input_data: TextInput):
    try:
        sql = f"UPDATE {EMBED_TABLE} SET text = $1 WHERE id = $2;"
        await execute_query(sql, [input_data.text, row_id])
        return {"message": "Embedding updated successfully"}
    except Exception as e:
        logger.error(f"Error updating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error updating embedding")

async def delete_embedding(row_id: int):
    try:
        sql = f"DELETE FROM {EMBED_TABLE} WHERE id = $1;"
        await execute_query(sql, [row_id])
        return {"message": "Embedding deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting embedding: {e}")
        raise HTTPException(status_code=500, detail="Error deleting embedding")