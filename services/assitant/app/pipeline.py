# -*- coding: utf-8 -*-
# app/pipeline.py
from app.logger import get_logger
from app.embed import get_model
from app.chromadb import get_chroma_client, chroma_list_collections, chroma_create_collection


logger = get_logger(__name__)


# Pipeline for initializing and managing the application state

async def initialize_pipeline(preload_models: bool = False):
    """
    Initialize the application pipeline, including model loading and ChromaDB setup.
    
    Args:
        preload_models (bool): Whether to preload models at startup.
    """
    try:
        # Load embedding model if required
        if preload_models:
            model = await get_model()
            logger.info(f"Embedding model loaded successfully: {model}")
        else:
            logger.info("Model preloading is disabled, skipping model loading.")

        # Initialize ChromaDB client
        client = get_chroma_client()
        collections = await chroma_list_collections()
        logger.info(f"ChromaDB collections loaded: {collections}")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

# pipeline get collections: embed, chroma
async def get_collections():
    """
    Retrieve the list of collections from ChromaDB.
    
    Returns:
        List of collection names.
    """
    try:
        client = get_chroma_client()
        collections = await chroma_list_collections()
        logger.info(f"Retrieved collections: {collections}")
        return collections
    except Exception as e:
        logger.error(f"Failed to retrieve collections: {e}")
        raise 
