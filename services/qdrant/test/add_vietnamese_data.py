#!/usr/bin/env python3
"""
Script để thêm dữ liệu tiếng Việt vào Qdrant
Sử dụng file luật kế toán Việt Nam 2015 làm dữ liệu mẫu
"""

import sys
import os
import json
import re
import asyncio
from typing import List, Dict, Any

# Thêm path để import modules
sys.path.append('src')

from qdrant_mcp.embedding.models.nomic import NomicEmbeddingModel
import requests


def read_vietnamese_law_file() -> str:
    """Đọc file luật kế toán Việt Nam"""
    file_path = "../../docs/luat_ke_toan_vietnam_2015.md"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Đã đọc file thành công: {len(content)} ký tự")
        return content
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return ""


def split_content_into_chunks(content: str, max_length: int = 500) -> List[Dict[str, Any]]:
    """Chia nội dung thành các chunk nhỏ"""
    # Chia theo đoạn văn
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    chunks = []
    for i, paragraph in enumerate(paragraphs):
        # Bỏ qua các dòng quá ngắn
        if len(paragraph) < 50:
            continue
            
        # Chia đoạn dài thành các chunk nhỏ hơn
        if len(paragraph) > max_length:
            sentences = re.split(r'[.!?]\s+', paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) > max_length and current_chunk:
                    chunks.append({
                        'id': len(chunks),
                        'text': current_chunk.strip(),
                        'metadata': {
                            'source': 'luat_ke_toan_vietnam_2015.md',
                            'chunk_type': 'paragraph_split',
                            'original_paragraph': i
                        }
                    })
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            if current_chunk.strip():
                chunks.append({
                    'id': len(chunks),
                    'text': current_chunk.strip(),
                    'metadata': {
                        'source': 'luat_ke_toan_vietnam_2015.md',
                        'chunk_type': 'paragraph_split',
                        'original_paragraph': i
                    }
                })
        else:
            chunks.append({
                'id': len(chunks),
                'text': paragraph,
                'metadata': {
                    'source': 'luat_ke_toan_vietnam_2015.md',
                    'chunk_type': 'full_paragraph',
                    'original_paragraph': i
                }
            })
    
    return chunks


def create_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tạo embeddings cho các text chunks"""
    print("🔄 Đang tạo embeddings...")
    
    try:
        # Khởi tạo Nomic embedding model
        model = NomicEmbeddingModel()
        
        embedded_chunks = []
        
        async def process_chunks():
            await model.load_model()
            
            for i, chunk in enumerate(chunks):
                print(f"Đang xử lý chunk {i+1}/{len(chunks)}: {chunk['text'][:50]}...")
                
                # Tạo embedding
                embedding = await model.embed_text(chunk['text'], prompt_name="document")
                
                # Thêm embedding vào chunk
                chunk['embedding'] = embedding
                embedded_chunks.append(chunk)
                
                # Giới hạn số lượng để test
                if i >= 9:  # Chỉ xử lý 10 chunks đầu
                    break
        
        # Chạy async function
        asyncio.run(process_chunks())
        
        print(f"✅ Đã tạo embeddings cho {len(embedded_chunks)} chunks")
        return embedded_chunks
        
    except Exception as e:
        print(f"❌ Lỗi tạo embeddings: {e}")
        return []


def create_qdrant_collection(collection_name: str = "vietnamese_law") -> bool:
    """Tạo collection trong Qdrant"""
    url = "http://localhost:6333/collections/vietnamese_law"
    
    # Xóa collection cũ nếu có
    try:
        requests.delete(url)
        print("🗑️ Đã xóa collection cũ")
    except:
        pass
    
    # Tạo collection mới
    payload = {
        "vectors": {
            "size": 768,  # Kích thước vector của Nomic
            "distance": "Cosine"
        }
    }
    
    try:
        response = requests.put(url, json=payload)
        if response.status_code in [200, 201]:
            print(f"✅ Đã tạo collection '{collection_name}' thành công")
            return True
        else:
            print(f"❌ Lỗi tạo collection: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        return False


def upload_to_qdrant(embedded_chunks: List[Dict[str, Any]], collection_name: str = "vietnamese_law") -> bool:
    """Upload embeddings lên Qdrant"""
    if not embedded_chunks:
        print("❌ Không có dữ liệu để upload")
        return False
    
    url = f"http://localhost:6333/collections/{collection_name}/points"
    
    # Chuẩn bị dữ liệu theo format của Qdrant
    points = []
    for chunk in embedded_chunks:
        points.append({
            "id": chunk['id'],
            "vector": chunk['embedding'],
            "payload": {
                "text": chunk['text'],
                "metadata": chunk['metadata']
            }
        })
    
    payload = {"points": points}
    
    try:
        print(f"🚀 Đang upload {len(points)} điểm dữ liệu...")
        response = requests.put(url, json=payload)
        
        if response.status_code in [200, 201]:
            print(f"✅ Đã upload thành công {len(points)} điểm dữ liệu")
            return True
        else:
            print(f"❌ Lỗi upload: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        return False


def test_search(query: str = "kế toán trưởng", collection_name: str = "vietnamese_law") -> None:
    """Test tìm kiếm semantic"""
    print(f"\n🔍 Test tìm kiếm với query: '{query}'")
    
    try:
        # Tạo embedding cho query
        model = NomicEmbeddingModel()
        
        async def get_query_embedding():
            await model.load_model()
            return await model.embed_text(query, prompt_name="query")
        
        query_embedding = asyncio.run(get_query_embedding())
        
        # Tìm kiếm trong Qdrant
        url = f"http://localhost:6333/collections/{collection_name}/points/search"
        payload = {
            "vector": query_embedding,
            "limit": 3,
            "with_payload": True
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Tìm thấy {len(results['result'])} kết quả:")
            
            for i, result in enumerate(results['result']):
                score = result['score']
                text = result['payload']['text'][:200]
                print(f"\n{i+1}. Score: {score:.4f}")
                print(f"   Text: {text}...")
        else:
            print(f"❌ Lỗi tìm kiếm: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Lỗi test tìm kiếm: {e}")


def main():
    """Hàm chính"""
    print("🚀 Bắt đầu thêm dữ liệu tiếng Việt vào Qdrant")
    print("=" * 50)
    
    # 1. Đọc file
    content = read_vietnamese_law_file()
    if not content:
        return
    
    # 2. Chia nhỏ content
    print("\n📝 Chia nhỏ nội dung...")
    chunks = split_content_into_chunks(content)
    print(f"✅ Đã chia thành {len(chunks)} chunks")
    
    # 3. Tạo embeddings
    print("\n🧠 Tạo embeddings...")
    embedded_chunks = create_embeddings(chunks)
    if not embedded_chunks:
        return
    
    # 4. Tạo collection
    print("\n🗃️ Tạo Qdrant collection...")
    if not create_qdrant_collection():
        return
    
    # 5. Upload dữ liệu
    print("\n⬆️ Upload dữ liệu...")
    if not upload_to_qdrant(embedded_chunks):
        return
    
    # 6. Test tìm kiếm
    print("\n🔍 Test tìm kiếm...")
    test_search("kế toán trưởng")
    test_search("báo cáo tài chính")
    test_search("điều kiện kinh doanh")
    
    print("\n🎉 Hoàn thành!")


if __name__ == "__main__":
    main()
