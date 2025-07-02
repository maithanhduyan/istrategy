#!/usr/bin/env python3
"""Test script for embedding tools."""

import asyncio
import json
import sys
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_embedding_tools():
    """Test embedding tools functionality."""
    try:
        # Import after ensuring path
        sys.path.insert(0, "src")
        from qdrant_mcp.embedding.pipeline.text_processor import get_embedding_pipeline
        from qdrant_mcp.embedding.schemas import NomicTextInput, EmbeddingBatchInput
        
        print("🧪 Testing Embedding Pipeline")
        print("=" * 50)
        
        # Get pipeline
        pipeline = await get_embedding_pipeline()
        print("✓ Pipeline initialized")
        
        # Test 1: Single embedding
        print("\n1. Testing single text embedding...")
        input_data = NomicTextInput(
            text="What is the meaning of life?",
            prompt_name="search_query"
        )
        
        result = await pipeline.create_embedding(input_data, model_name="nomic")
        print(f"✓ Generated embedding with {result.dimensions} dimensions")
        print(f"  Text: {result.text}")
        print(f"  Model: {result.model}")
        print(f"  Prompt: {result.prompt_name}")
        print(f"  First 5 values: {result.embeddings[:5]}")
        
        # Test 2: Batch embeddings
        print("\n2. Testing batch embeddings...")
        batch_input = EmbeddingBatchInput(
            texts=[
                "Machine learning is transforming technology",
                "Natural language processing enables computers to understand text",
                "Vector databases store high-dimensional data"
            ],
            prompt_name="search_document"
        )
        
        batch_result = await pipeline.create_batch_embeddings(batch_input, model_name="nomic")
        print(f"✓ Generated {batch_result.count} embeddings with {batch_result.dimensions} dimensions each")
        print(f"  Model: {batch_result.model}")
        
        # Test 3: Model info
        print("\n3. Testing model info...")
        model_info = await pipeline.get_model_info("nomic")
        print(f"✓ Model info: {json.dumps(model_info, indent=2)}")
        
        # Test 4: Available models
        print("\n4. Testing available models...")
        models = pipeline.list_available_models()
        print(f"✓ Available models: {models}")
        
        print("\n🎉 All embedding tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration_with_qdrant():
    """Test embedding + Qdrant integration."""
    try:
        print("\n🧪 Testing Embedding + Qdrant Integration")
        print("=" * 50)
        
        # Import required modules
        sys.path.insert(0, "src")
        from qdrant_mcp.embedding.pipeline.text_processor import get_embedding_pipeline
        from qdrant_mcp.embedding.schemas import NomicTextInput
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        import uuid
        
        # Initialize
        pipeline = await get_embedding_pipeline()
        client = QdrantClient(host="localhost", port=6333)
        
        # Test data
        texts = [
            "Python is a versatile programming language",
            "Machine learning models process data to make predictions",
            "Vector databases enable semantic search capabilities"
        ]
        
        collection_name = "embedding_test"
        
        # Create embedding for first text to get dimensions
        sample_input = NomicTextInput(text=texts[0], prompt_name="search_document")
        sample_result = await pipeline.create_embedding(sample_input)
        vector_size = sample_result.dimensions
        
        print(f"✓ Sample embedding has {vector_size} dimensions")
        
        # Create collection
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✓ Created collection '{collection_name}'")
        
        # Embed and store all texts
        points = []
        for i, text in enumerate(texts):
            input_data = NomicTextInput(text=text, prompt_name="search_document")
            result = await pipeline.create_embedding(input_data)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=result.embeddings,
                payload={"text": text, "index": i}
            )
            points.append(point)
        
        client.upsert(collection_name=collection_name, points=points)
        print(f"✓ Stored {len(points)} embeddings in Qdrant")
        
        # Test search
        query_text = "programming languages and development"
        query_input = NomicTextInput(text=query_text, prompt_name="search_query")
        query_result = await pipeline.create_embedding(query_input)
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_result.embeddings,
            limit=3
        )
        
        print(f"✓ Search results for '{query_text}':")
        for hit in search_result:
            print(f"  - Score: {hit.score:.3f}, Text: {hit.payload['text']}")
        
        print("\n🎉 Integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    print("🚀 Starting Embedding Tools Tests")
    print("=" * 60)
    
    # Test 1: Basic embedding functionality
    success1 = await test_embedding_tools()
    
    # Test 2: Integration with Qdrant
    success2 = await test_integration_with_qdrant()
    
    # Summary
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)
