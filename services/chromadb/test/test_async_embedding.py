#!/usr/bin/env python3
"""Test script để kiểm tra async preload embedding function."""

import sys
import time
import asyncio
sys.path.insert(0, 'src/chromadb-mcp')

from embedding import NomicEmbeddingFunction, _nomic_model, _nomic_model_ready, preload_nomic_model_async

async def test_async_embedding():
    """Test async embedding function performance."""
    print("=== Test Async Preload Embedding Function ===")
    
    # Kiểm tra trạng thái model
    print(f"Model preloaded: {_nomic_model is not None}")
    print(f"Async ready: {_nomic_model_ready.is_set()}")
    
    # Test tạo embedding function với các prompt khác nhau
    embeddings = {
        "default": NomicEmbeddingFunction(),
        "query": NomicEmbeddingFunction(prompt_name="query"),
        "document": NomicEmbeddingFunction(prompt_name="document"),
    }
    
    # Test documents tiếng Việt và Anh
    test_docs = [
        "Xin chào, đây là tài liệu tiếng Việt về AI.",
        "Hello, this is an English document about artificial intelligence.",
        "ChromaDB là một vector database mạnh mẽ."
    ]
    
    print("\n=== Testing Embedding Generation ===")
    for name, embedding_func in embeddings.items():
        start_time = time.time()
        result = embedding_func(test_docs)
        end_time = time.time()
        
        print(f"{name}: Generated {len(result)} embeddings in {end_time - start_time:.3f}s")
        print(f"  - First embedding dimensions: {len(result[0])}")
        print(f"  - Sample values: {result[0][:5]}")
    
    print("\n=== Async Preload Test ===")
    if not _nomic_model_ready.is_set():
        print("Starting async preload...")
        start_time = time.time()
        await preload_nomic_model_async()
        end_time = time.time()
        print(f"Async preload completed in {end_time - start_time:.3f}s")
        print(f"Async ready: {_nomic_model_ready.is_set()}")
    else:
        print("Async preload already completed")

if __name__ == "__main__":
    asyncio.run(test_async_embedding())
