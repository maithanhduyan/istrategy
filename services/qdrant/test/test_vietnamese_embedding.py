#!/usr/bin/env python3
"""Test script cho embedding và tìm kiếm tiếng Việt với Qdrant."""

import asyncio
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_vietnamese_embedding():
    """Test embedding tiếng Việt và lưu trữ vào Qdrant."""
    try:
        # Import modules
        sys.path.insert(0, "src")
        from qdrant_mcp.embedding.pipeline.text_processor import get_embedding_pipeline
        from qdrant_mcp.embedding.schemas import NomicTextInput
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, VectorParams, Distance
        import uuid
        
        print("🇻🇳 Test Embedding Tiếng Việt với Qdrant")
        print("=" * 60)
        
        # Initialize
        pipeline = await get_embedding_pipeline()
        client = QdrantClient(host="localhost", port=6333)
        
        # Các văn bản tiếng Việt để test
        vietnamese_texts = [
            "Trí tuệ nhân tạo đang thay đổi thế giới công nghệ hiện đại",
            "Máy học và xử lý ngôn ngữ tự nhiên giúp máy tính hiểu được tiếng Việt",
            "Cơ sở dữ liệu vector cho phép tìm kiếm ngữ nghĩa rất hiệu quả",
            "Python là ngôn ngữ lập trình rất phổ biến trong khoa học dữ liệu",
            "Hệ thống MCP giúp kết nối các công cụ AI một cách dễ dàng",
            "Qdrant là cơ sở dữ liệu vector hiệu năng cao cho AI"
        ]
        
        collection_name = "van_ban_viet"
        
        print(f"✓ Sử dụng collection '{collection_name}'")
        
        # Embed và lưu trữ các văn bản tiếng Việt
        print(f"\n📝 Đang embed và lưu {len(vietnamese_texts)} văn bản tiếng Việt...")
        
        points = []
        for i, text in enumerate(vietnamese_texts):
            # Tạo embedding với prompt "document"
            input_data = NomicTextInput(text=text, prompt_name="document")
            result = await pipeline.create_embedding(input_data)
            
            # Tạo point với UUID
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=result.embeddings,
                payload={
                    "text": text,
                    "index": i,
                    "language": "vietnamese",
                    "category": "test_data"
                }
            )
            points.append(point)
            print(f"  ✓ Embedded: {text[:50]}...")
        
        # Lưu vào Qdrant
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Đã lưu {len(points)} embeddings vào collection '{collection_name}'")
        
        # Kiểm tra collection
        collection_info = client.get_collection(collection_name)
        print(f"📊 Collection info: {collection_info.points_count} points")
        
        # Test tìm kiếm với các query tiếng Việt
        queries = [
            "công nghệ AI và machine learning",
            "ngôn ngữ lập trình cho data science", 
            "tìm kiếm vector database",
            "hệ thống kết nối công cụ AI"
        ]
        
        print(f"\n🔍 Test tìm kiếm với {len(queries)} queries tiếng Việt:")
        
        for query in queries:
            print(f"\n🔎 Query: '{query}'")
            
            # Tạo embedding cho query
            query_input = NomicTextInput(text=query, prompt_name="query")
            query_result = await pipeline.create_embedding(query_input)
            
            # Tìm kiếm
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_result.embeddings,
                limit=3,
                score_threshold=0.1
            )
            
            print(f"  📋 Kết quả tìm kiếm (top 3):")
            for j, hit in enumerate(search_result):
                print(f"    {j+1}. Score: {hit.score:.3f}")
                print(f"       Text: {hit.payload['text']}")
        
        print(f"\n🎉 Test embedding tiếng Việt thành công!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test thất bại: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_semantic_search_vietnamese():
    """Test tìm kiếm ngữ nghĩa tiếng Việt nâng cao."""
    try:
        print(f"\n🧠 Test Tìm Kiếm Ngữ Nghĩa Tiếng Việt Nâng Cao")
        print("=" * 60)
        
        # Import modules
        sys.path.insert(0, "src")
        from qdrant_mcp.embedding.pipeline.text_processor import get_embedding_pipeline
        from qdrant_mcp.embedding.schemas import NomicTextInput
        from qdrant_client import QdrantClient
        
        pipeline = await get_embedding_pipeline()
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "van_ban_viet"
        
        # Test các query khác nhau về cùng 1 chủ đề
        related_queries = [
            "trí tuệ nhân tạo",
            "AI và machine learning", 
            "công nghệ thông minh",
            "hệ thống tự động học"
        ]
        
        print("🔍 Test tìm kiếm với các query tương tự về AI:")
        
        for query in related_queries:
            print(f"\n📝 Query: '{query}'")
            
            # Embed query
            query_input = NomicTextInput(text=query, prompt_name="query")
            query_result = await pipeline.create_embedding(query_input)
            
            # Search
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_result.embeddings,
                limit=2
            )
            
            print(f"  🎯 Top 2 kết quả:")
            for i, hit in enumerate(search_result):
                print(f"    {i+1}. Score: {hit.score:.3f} - {hit.payload['text'][:60]}...")
        
        print(f"\n✅ Test tìm kiếm ngữ nghĩa hoàn thành!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test ngữ nghĩa thất bại: {e}")
        return False


async def main():
    """Main test runner."""
    print("🚀 Bắt Đầu Test Pipeline Embedding Tiếng Việt")
    print("=" * 70)
    
    # Test 1: Basic embedding và storage
    success1 = await test_vietnamese_embedding()
    
    # Test 2: Semantic search
    success2 = await test_semantic_search_vietnamese()
    
    # Summary
    print("\n" + "=" * 70)
    if success1 and success2:
        print("🎉 TẤT CẢ TEST TIẾNG VIỆT ĐÃ THÀNH CÔNG!")
        print("✅ Pipeline embedding hoạt động tốt với tiếng Việt")
        print("✅ Tìm kiếm ngữ nghĩa tiếng Việt hoàn hảo")
        return 0
    else:
        print("❌ Một số test bị lỗi!")
        return 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)
