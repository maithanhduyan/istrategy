#!/usr/bin/env python3
"""
Script test tìm kiếm semantic tiếng Việt trong Qdrant
Sử dụng MCP embedding tools
"""

import sys
import asyncio
import requests

# Thêm path để import modules
sys.path.append('src')

from qdrant_mcp.embedding.models.nomic import NomicEmbeddingModel


async def test_semantic_search():
    """Test tìm kiếm semantic với các query tiếng Việt"""
    
    # Các query test
    queries = [
        "kế toán trưởng có trách nhiệm gì",
        "điều kiện để làm kế toán viên",
        "báo cáo tài chính phải làm như thế nào",
        "doanh nghiệp cần tuân thủ quy định gì",
        "quyền và nghĩa vụ của người làm kế toán"
    ]
    
    print("🔍 Testing Semantic Search với dữ liệu tiếng Việt")
    print("=" * 60)
    
    # Khởi tạo model
    model = NomicEmbeddingModel()
    await model.load_model()
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Tạo embedding cho query
            query_embedding = await model.embed_text(query, prompt_name="query")
            
            # Tìm kiếm trong Qdrant
            url = "http://localhost:6333/collections/vietnamese_law/points/search"
            payload = {
                "vector": query_embedding,
                "limit": 2,
                "with_payload": True
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                results = response.json()
                
                if results['result']:
                    print(f"✅ Tìm thấy {len(results['result'])} kết quả:")
                    
                    for j, result in enumerate(results['result'], 1):
                        score = result['score']
                        text = result['payload']['text']
                        source = result['payload']['metadata']['source']
                        
                        print(f"\n   {j}. Score: {score:.4f}")
                        print(f"      Source: {source}")
                        print(f"      Text: {text[:300]}...")
                        
                        if len(text) > 300:
                            print(f"             (và {len(text) - 300} ký tự nữa)")
                else:
                    print("❌ Không tìm thấy kết quả nào")
                    
            else:
                print(f"❌ Lỗi tìm kiếm: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Lỗi xử lý query: {e}")
    
    print("\n🎉 Hoàn thành test semantic search!")


async def get_collection_stats():
    """Lấy thống kê collection"""
    print("\n📊 Thống kê Collection 'vietnamese_law'")
    print("-" * 40)
    
    try:
        url = "http://localhost:6333/collections/vietnamese_law"
        response = requests.get(url)
        
        if response.status_code == 200:
            info = response.json()
            result = info['result']
            
            print(f"✅ Tên collection: {result['config']['params']['vectors']['']['size']} dimensions")
            print(f"✅ Số điểm dữ liệu: {result['points_count']}")
            print(f"✅ Vector distance: {result['config']['params']['vectors']['']['distance']}")
            print(f"✅ Trạng thái: {result['status']}")
        else:
            print(f"❌ Lỗi lấy thông tin: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")


async def main():
    """Hàm chính"""
    print("🚀 Bắt đầu test Qdrant với dữ liệu tiếng Việt")
    
    # Lấy thống kê collection
    await get_collection_stats()
    
    # Test semantic search
    await test_semantic_search()


if __name__ == "__main__":
    asyncio.run(main())
