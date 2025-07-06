# -*- coding: utf-8 -*-
# app/pipeline.py
from typing import List, Dict, Any, Optional
from app.logger import get_logger
from app.embed import get_model, create_embedding, TextInput
from app.chromadb import (
    get_chroma_client, chroma_list_collections, chroma_create_collection,
    chroma_add_documents, chroma_query_documents, chroma_get_documents,
    chroma_delete_collection, chroma_get_collection_stats,
    chroma_update_documents, chroma_delete_documents
)


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
            model = get_model()
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


# === PIPELINE FUNCTIONS FOR CHROMADB + EMBEDDING ===

async def create_collection_with_embeddings(
    collection_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new ChromaDB collection optimized for embeddings.
    
    Args:
        collection_name: Name of the collection to create
        metadata: Optional metadata for the collection
    
    Returns:
        Dictionary containing creation result and collection info
    """
    try:
        if metadata is None:
            metadata = {
                "description": f"Embedding collection: {collection_name}",
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "created_by": "pipeline"
            }
        
        result = await chroma_create_collection(
            collection_name=collection_name,
            metadata=metadata
        )
        
        logger.info(f"Created embedding collection '{collection_name}' successfully")
        return {
            "status": "success",
            "collection_name": collection_name,
            "result": result,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Failed to create embedding collection '{collection_name}': {e}")
        raise


async def add_text_documents_with_embeddings(
    collection_name: str,
    texts: List[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Add text documents to ChromaDB collection. ChromaDB will auto-generate embeddings.
    
    Args:
        collection_name: Name of the collection
        texts: List of text documents to add
        ids: Optional list of document IDs (auto-generated if None)
        metadatas: Optional list of metadata for each document
    
    Returns:
        Dictionary containing operation result
    """
    try:
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{hash(text) % 10000}" for i, text in enumerate(texts)]
        
        # Add default metadata if not provided
        if metadatas is None:
            metadatas = [{"source": "pipeline", "length": len(text)} for text in texts]
        
        # Add documents to ChromaDB (ChromaDB will generate embeddings automatically)
        result = await chroma_add_documents(
            collection_name=collection_name,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(texts)} documents to '{collection_name}'")
        return {
            "status": "success",
            "collection_name": collection_name,
            "documents_added": len(texts),
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to add documents to '{collection_name}': {e}")
        raise


async def search_similar_documents(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search for similar documents using text query. ChromaDB will generate embeddings automatically.
    
    Args:
        collection_name: Name of the collection to search
        query_text: Text query to search for
        n_results: Number of results to return
        where: Optional metadata filter
    
    Returns:
        Dictionary containing search results
    """
    try:
        # Query ChromaDB (ChromaDB will generate embeddings automatically)
        result = await chroma_query_documents(
            collection_name=collection_name,
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        logger.info(f"Found {len(result.get('documents', [[]])[0])} similar documents for query")
        return {
            "status": "success",
            "collection_name": collection_name,
            "query_text": query_text,
            "n_results": n_results,
            "results": result
        }
    except Exception as e:
        logger.error(f"Failed to search similar documents in '{collection_name}': {e}")
        raise


async def get_collection_summary(collection_name: str) -> Dict[str, Any]:
    """
    Get comprehensive summary of a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection
    
    Returns:
        Dictionary containing collection summary
    """
    try:
        stats = await chroma_get_collection_stats(collection_name)
        
        # Get sample documents
        sample_docs = await chroma_get_documents(
            collection_name=collection_name,
            limit=3
        )
        
        summary = {
            "status": "success",
            "collection_name": collection_name,
            "stats": stats,
            "sample_documents": sample_docs,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
        }
        
        logger.info(f"Generated summary for collection '{collection_name}'")
        return summary
    except Exception as e:
        logger.error(f"Failed to get summary for collection '{collection_name}': {e}")
        raise


async def update_document_with_new_text(
    collection_name: str,
    document_id: str,
    new_text: str,
    new_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update a document with new text. ChromaDB will regenerate embeddings automatically.
    
    Args:
        collection_name: Name of the collection
        document_id: ID of the document to update
        new_text: New text content
        new_metadata: Optional new metadata
    
    Returns:
        Dictionary containing update result
    """
    try:
        # Update metadata with text length
        if new_metadata is None:
            new_metadata = {}
        new_metadata.update({
            "length": len(new_text),
            "updated_by": "pipeline"
        })
        
        # Update document in ChromaDB (ChromaDB will generate embeddings automatically)
        result = await chroma_update_documents(
            collection_name=collection_name,
            ids=[document_id],
            documents=[new_text],
            metadatas=[new_metadata]
        )
        
        logger.info(f"Updated document '{document_id}' with new text")
        return {
            "status": "success",
            "collection_name": collection_name,
            "document_id": document_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to update document '{document_id}' in '{collection_name}': {e}")
        raise


async def delete_collection_safely(collection_name: str) -> Dict[str, Any]:
    """
    Safely delete a ChromaDB collection with confirmation.
    
    Args:
        collection_name: Name of the collection to delete
    
    Returns:
        Dictionary containing deletion result
    """
    try:
        # Get collection stats before deletion
        stats = await chroma_get_collection_stats(collection_name)
        
        # Delete the collection
        result = await chroma_delete_collection(collection_name)
        
        logger.info(f"Safely deleted collection '{collection_name}' with {stats.get('count', 0)} documents")
        return {
            "status": "success",
            "collection_name": collection_name,
            "documents_deleted": stats.get("count", 0),
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to delete collection '{collection_name}': {e}")
        raise


async def batch_process_texts(
    texts: List[str],
    collection_name: Optional[str] = None,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Process a large batch of texts efficiently. If collection specified, stores with auto-generated embeddings.
    
    Args:
        texts: List of texts to process
        collection_name: Optional collection to store results
        batch_size: Number of texts to process per batch
    
    Returns:
        Dictionary containing batch processing results
    """
    try:
        total_texts = len(texts)
        processed_count = 0
        
        # Process in batches
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Store in collection if specified (ChromaDB will generate embeddings automatically)
            if collection_name:
                batch_ids = [f"batch_{i // batch_size}_doc_{j}" for j in range(len(batch_texts))]
                batch_metadatas = [{"batch": i // batch_size, "index": j} for j in range(len(batch_texts))]
                
                await chroma_add_documents(
                    collection_name=collection_name,
                    documents=batch_texts,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
            
            processed_count += len(batch_texts)
            logger.info(f"Processed batch {i // batch_size + 1}, total: {processed_count}/{total_texts}")
        
        result = {
            "status": "success",
            "total_texts": total_texts,
            "processed_count": processed_count,
            "batch_size": batch_size,
            "batches_processed": (total_texts + batch_size - 1) // batch_size
        }
        
        if collection_name:
            result["collection_name"] = collection_name
        
        logger.info(f"Completed batch processing of {total_texts} texts")
        return result
    except Exception as e:
        logger.error(f"Failed to batch process texts: {e}")
        raise 
