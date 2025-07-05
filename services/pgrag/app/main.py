import sys
from contextlib import asynccontextmanager
from db import get_pool, close_pool, init_tables, install_pgvector_extension, create_database_if_not_exists
from logger import get_logger, stop_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import asyncpg
from embed import (
    get_model, create_embedding, save_embedding, get_embedding,
    search_embeddings, update_embedding, delete_embedding, TextInput
)

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    try:
        # Create database if not exists (for Docker environment)
        create_database_if_not_exists()
        
        await install_pgvector_extension()
        await init_tables()
        logger.info("Database initialized successfully")
        
        # Load embedding model
        model = get_model()
        logger.info(f"Embedding model loaded successfully: {model}")
        
    except Exception as e:
        logger.error(f"Failed to initialize database or load model: {e}")
    
    yield
    
    # Shutdown
    try:
        await close_pool()
        logger.info("Database pool closed successfully")
    except Exception as e:
        logger.error(f"Failed to close database pool: {e}")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/embeddings/create")
async def create_text_embedding(input_data: TextInput):
    """Create embeddings for input text."""
    embeddings = await create_embedding(input_data)
    return {"text": input_data.text, "embeddings": embeddings}


@app.post("/embeddings/save")
async def save_text_embedding(input_data: TextInput):
    """Save text embeddings to database."""
    embedding_id = await save_embedding(input_data)
    return {"id": embedding_id, "text": input_data.text, "message": "Embedding saved successfully"}


@app.get("/embeddings/{row_id}")
async def get_text_embedding(row_id: int):
    """Get embedding by ID."""
    return await get_embedding(row_id)


@app.get("/embeddings/search/{query}")
async def search_text_embeddings(query: str):
    """Search embeddings by text query."""
    return await search_embeddings(query)


@app.put("/embeddings/{row_id}")
async def update_text_embedding(row_id: int, input_data: TextInput):
    """Update embedding by ID."""
    return await update_embedding(row_id, input_data)


@app.delete("/embeddings/{row_id}")
async def delete_text_embedding(row_id: int):
    """Delete embedding by ID."""
    return await delete_embedding(row_id)

    
def main():
    """Run the FastAPI application."""
    import uvicorn
    import os
    
    # Get host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

    

if __name__ == "__main__":
    main()