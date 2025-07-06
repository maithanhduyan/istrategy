#
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.logger import get_logger, LOGGING_CONFIG
import asyncio
from app.router import router as app_router
from app.mcp import router as mcp_router
from app.db import init_database    

logger = get_logger(__name__)

APP_RELOAD = False  # Set to True to enable auto-reload in development

# Waiting for loading embed and chromadb modules
from app.embed import get_model
from app.chromadb import get_chroma_client, chroma_list_collections, chroma_create_collection


_client= None  # Global variable to hold the ChromaDB client instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # App Startup
    try:
        # Initialize database
        await asyncio.to_thread(init_database)
        logger.info("Database initialized successfully")
        
        # Initialize ChromaDB client
        try:
            await asyncio.to_thread(get_chroma_client)
            logger.info("ChromaDB client initialized successfully")
            collections = await chroma_list_collections(10)
            logger.info(f"ChromaDB client collections: {collections}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}. Continuing without ChromaDB.")

        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
    
    yield
    
    # App Shutdown
    try:
        logger.info("Application shutting down")
    except Exception as e:
        logger.error(f"Failed to shut down application: {e}")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include authentication and API routes
app.include_router(app_router, prefix="/api", tags=["api"])
app.include_router(mcp_router, prefix="/mcp", tags=["mcp"])

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

def main():
    """Main entry point for the application."""
    import os
    import uvicorn
    # Load environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(app, host=host, port=port,log_config=LOGGING_CONFIG, reload=APP_RELOAD)

if __name__ == "__main__":
    main()