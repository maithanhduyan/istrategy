# Demo tìm kiếm nhanh với FastAPI và pgvector
# Sử dụng pgvector để lưu trữ và truy vấn vector embedding trong PostgreSQL.
import logging
import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.embed import get_model
import app.db

# Preload model
MODEL= get_model()

# Database configuration
db_pool = app.db.get_pool()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asyncpg pool
async def get_pool():
    if not hasattr(get_pool, "_pool"):
        get_pool._pool = await asyncpg.create_pool(
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            min_size=1,
            max_size=5,
        )
    return get_pool._pool

async def close_pool():
    if hasattr(get_pool, "_pool"):
        await get_pool._pool.close()
        del get_pool._pool

async def setup_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Cài extension pgvector nếu chưa có
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        sql = f"""
        CREATE TABLE IF NOT EXISTS {EMBED_TABLE} (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR(384)
        );
        """
        await conn.execute(sql)

class TextInput(BaseModel):
    """Input model for text to be embedded."""
    text: str

async def embed_text(text: str) -> List[float]:
    """Generate text embeddings using sentence-transformers."""
    try:
        embeddings = MODEL.encode(text).tolist()
        logging.info(f"Generated embeddings for text: {text[:50]}...")
        return embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Error generating embeddings")

async def save_embedding_to_db(text: str, embeddings: List[float]) -> int:
    """Save the generated embeddings to the database."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            sql = f"INSERT INTO {EMBED_TABLE} (text, embedding) VALUES ($1, $2) RETURNING id;"
            logging.info(f"Saving embeddings for text: {text[:50]}... with {len(embeddings)} dimensions")
            row = await conn.fetchrow(sql, text, embeddings)
            return row["id"]
    except Exception as e:
        logging.error(f"Error saving embeddings to database: {e}")
        raise HTTPException(status_code=500, detail="Error saving embeddings to database")

async def get_embedding_by_id(row_id: int) -> Optional[dict]:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            sql = f"SELECT id, text, embedding FROM {EMBED_TABLE} WHERE id = $1;"
            row = await conn.fetchrow(sql, row_id)
            if row:
                return {"id": row["id"], "text": row["text"], "embedding": row["embedding"]}
            return None
    except Exception as e:
        logging.error(f"Error fetching embedding: {e}")
        raise HTTPException(status_code=500, detail="Error fetching embedding")

async def query_by_text(text: str) -> List[dict]:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            sql = f"SELECT id, text FROM {EMBED_TABLE} WHERE text ILIKE $1;"
            rows = await conn.fetch(sql, f"%{text}%")
            return [{"id": row["id"], "text": row["text"]} for row in rows]
    except Exception as e:
        logging.error(f"Error searching embeddings: {e}")
        raise HTTPException(status_code=500, detail="Error searching embeddings")

async def update_text(row_id: int, new_text: str):
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            sql = f"UPDATE {EMBED_TABLE} SET text = $1 WHERE id = $2;"
            await conn.execute(sql, new_text, row_id)
    except Exception as e:
        logging.error(f"Error updating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error updating embedding")

async def delete_row(row_id: int):
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            sql = f"DELETE FROM {EMBED_TABLE} WHERE id = $1;"
            await conn.execute(sql, row_id)
    except Exception as e:
        logging.error(f"Error deleting embedding: {e}")
        raise HTTPException(status_code=500, detail="Error deleting embedding")

app = FastAPI(
    title="Text Embedding Service",
    description="HTTP service for generating text embeddings",
    version="0.1.0"
)

@app.on_event("startup")
async def on_startup():
    await setup_table()

@app.on_event("shutdown")
async def on_shutdown():
    await close_pool()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Text Embedding Service is running", "version": "0.1.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/embed", response_model=List[float])
async def create_embedding(input_data: TextInput):
    """Create text embeddings using sentence-transformers."""
    embeddings = await embed_text(input_data.text)
    return embeddings

@app.post("/save", response_model=int)
async def save_embedding(input_data: TextInput):
    """Save text embeddings to the database."""
    embeddings = await embed_text(input_data.text)
    row_id = await save_embedding_to_db(input_data.text, embeddings)
    return row_id

@app.get("/embedding/{row_id}", response_model=dict)
async def get_embedding(row_id: int):
    """Get embedding by row ID."""
    result = await get_embedding_by_id(row_id)
    if not result:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return result

@app.get("/search", response_model=List[dict])
async def search_embeddings(query: str):
    """Search embeddings by text."""
    results = await query_by_text(query)
    if not results:
        raise HTTPException(status_code=404, detail="No embeddings found")
    return results

@app.put("/update/{row_id}", response_model=dict)
async def update_embedding(row_id: int, input_data: TextInput):
    """Update text embedding by row ID."""
    await update_text(row_id, input_data.text)
    return {"message": "Embedding updated successfully"}

@app.delete("/delete/{row_id}", response_model=dict)
async def delete_embedding(row_id: int):
    """Delete text embedding by row ID."""
    await delete_row(row_id)
    return {"message": "Embedding deleted successfully"}

def main():
    import uvicorn
    uvicorn.run("example.demo_fastapi_search:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
    # Chạy FastAPI server: uv run demo_fastapi_search.py
    # Gửi POST request đến /search với JSON: {"query": "your search text"}
    # Ví dụ: curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "hello"}'