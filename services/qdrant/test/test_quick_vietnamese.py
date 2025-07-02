#!/usr/bin/env python3
"""Test script ngắn gọn cho embedding tiếng Việt."""

import asyncio
import sys

async def test_quick_vietnamese():
    """Test nhanh embedding tiếng Việt."""
    sys.path.insert(0, "src")
    from qdrant_mcp.embedding.pipeline.text_processor import get_embedding_pipeline
    from qdrant_mcp.embedding.schemas import NomicTextInput
    from qdrant_client import QdrantClient
    
    print("🚀 Test Nhanh Embedding Tiếng Việt")
    print("=" * 40)
    
    # Init
    pipeline = await get_embedding_pipeline()
    client = QdrantClient(host="localhost", port=6333)
    
    # Test 1: Tạo embedding đơn giản
    print("\n📝 Tạo embedding cho câu tiếng Việt...")
    text = "Hôm nay trời đẹp quá!"
    input_data = NomicTextInput(text=text, prompt_name="document")
    result = await pipeline.create_embedding(input_data)
    
    print(f"✓ Text: {result.text}")
    print(f"✓ Dimensions: {result.dimensions}")
    print(f"✓ Model: {result.model}")
    print(f"✓ First 10 values: {result.embeddings[:10]}")
    
    # Test 2: Tìm kiếm trong collection tiếng Việt
    print(f"\n🔍 Tìm kiếm tương tự trong collection 'van_ban_viet'...")
    query_text = "Python programming"
    query_input = NomicTextInput(text=query_text, prompt_name="query")
    query_result = await pipeline.create_embedding(query_input)
    
    search_result = client.search(
        collection_name="van_ban_viet",
        query_vector=query_result.embeddings,
        limit=2
    )
    
    print(f"🔎 Query: '{query_text}'")
    print("📋 Kết quả:")
    for i, hit in enumerate(search_result):
        print(f"  {i+1}. Score: {hit.score:.3f}")
        print(f"     Text: {hit.payload['text']}")
    
    print("\n🎉 Test hoàn thành!")

if __name__ == "__main__":
    asyncio.run(test_quick_vietnamese())
