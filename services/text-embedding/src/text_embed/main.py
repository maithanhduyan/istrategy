"""Main FastAPI application for text embedding service."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Embedding Service",
    description="HTTP service for generating text embeddings",
    version="0.1.0"
)

class TextInput(BaseModel):
    """Input model for text to be embedded."""
    text: str

class NomicTextInput(BaseModel):
    """Input model for text to be embedded with Nomic model."""
    text: str
    prompt_name: str = "passage"  # Default prompt name for Nomic model

class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    text: str
    embeddings: List[float]
    model: str

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Text Embedding Service is running", "version": "0.1.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(input_data: TextInput):
    """Create text embeddings using sentence-transformers."""
    try:
        # Import here to avoid loading model at startup
        from sentence_transformers import SentenceTransformer
        
        # Load model (you can change this to other models)
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        embeddings = model.encode(input_data.text).tolist()
        
        logger.info(f"Generated embeddings for text: {input_data.text[:50]}...")
        
        return EmbeddingResponse(
            text=input_data.text,
            embeddings=embeddings,
            model=model_name
        )
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/embed/nomic", response_model=EmbeddingResponse)
async def create_nomic_embedding(input_data: NomicTextInput):
    """Create text embeddings using Nomic AI model."""
    try:
        # Import here to avoid loading model at startup
        from sentence_transformers import SentenceTransformer
        
        # Load Nomic model
        model_name = "nomic-ai/nomic-embed-text-v2-moe"
        model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Generate embeddings with prompt_name parameter
        embeddings = model.encode(input_data.text, prompt_name=input_data.prompt_name).tolist()
        
        logger.info(f"Generated Nomic embeddings for text: {input_data.text[:50]}... with prompt: {input_data.prompt_name}")
        
        return EmbeddingResponse(
            text=input_data.text,
            embeddings=embeddings,
            model=model_name
        )
    
    except Exception as e:
        logger.error(f"Error generating Nomic embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating Nomic embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
