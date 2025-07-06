# -*- coding: utf-8 -*-
# test/test_pipeline.py
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import (
    initialize_pipeline,
    get_collections,
    create_collection_with_embeddings,
    add_text_documents_with_embeddings,
    search_similar_documents,
    get_collection_summary
)

async def test_pipeline():
    """Test the complete ChromaDB + Embedding pipeline."""
    try:
        print("=== TESTING CHROMADB + EMBEDDING PIPELINE ===")
        
        # 1. Initialize pipeline
        print("\n1. Initializing pipeline...")
        await initialize_pipeline(preload_models=True)
        
        # 2. Get collections
        print("\n2. Getting existing collections...")
        collections = await get_collections()
        print(f"Existing collections: {collections}")
        
        # 3. Create new collection
        print("\n3. Creating new collection with embeddings...")
        import time
        collection_name = f"pipeline_test_collection_{int(time.time())}"
        create_result = await create_collection_with_embeddings(
            collection_name=collection_name,
            metadata={"description": "Test collection for pipeline", "test": True}
        )
        print(f"Create result: {create_result}")
        
        # 4. Add sample documents
        print("\n4. Adding sample documents with embeddings...")
        sample_texts = [
            "Artificial intelligence is transforming the world",
            "Machine learning algorithms improve with data",
            "Neural networks mimic the human brain",
            "Deep learning enables complex pattern recognition",
            "Natural language processing helps computers understand text"
        ]
        
        add_result = await add_text_documents_with_embeddings(
            collection_name=collection_name,
            texts=sample_texts
        )
        print(f"Add documents result: {add_result}")
        
        # 5. Search for similar documents
        print("\n5. Searching for similar documents...")
        query = "AI and machine learning technologies"
        search_result = await search_similar_documents(
            collection_name=collection_name,
            query_text=query,
            n_results=3
        )
        print(f"Search result for '{query}':")
        print(f"Found {len(search_result['results']['documents'][0])} similar documents")
        
        # 6. Get collection summary
        print("\n6. Getting collection summary...")
        summary = await get_collection_summary(collection_name)
        print(f"Collection summary: {summary}")
        
        print("\n=== PIPELINE TEST COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pipeline())
