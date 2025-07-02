#!/usr/bin/env python3
"""
Script test tra cứu luật kế toán trực tiếp từ Qdrant database
"""

import sys
import asyncio
import requests
import json
from typing import List, Dict

# Thêm path để import modules
sys.path.append('src')

from qdrant_mcp.embedding.models.nomic import NomicEmbeddingModel


class VietnameseLawSearcher:
    """Class để tra cứu luật kế toán Việt Nam từ Qdrant"""
    
    def __init__(self, collection_name: str = "vietnamese_law"):
        self.collection_name = collection_name
        self.qdrant_url = "http://localhost:6333"
        self.model = None
    
    async def initialize_model(self):
        """Khởi tạo model embedding"""
        if self.model is None:
            self.model = NomicEmbeddingModel()
            await self.model.load_model()
            print("✅ Đã khởi tạo model embedding")
    
    async def search_law(self, query: str, limit: int = 3) -> Dict:
        """Tìm kiếm thông tin luật kế toán"""
        await self.initialize_model()
        
        # Tạo embedding cho query
        query_embedding = await self.model.embed_text(query, prompt_name="query")
        
        # Gửi request tìm kiếm đến Qdrant
        search_url = f"{self.qdrant_url}/collections/{self.collection_name}/points/search"
        payload = {
            "vector": query_embedding,
            "limit": limit,
            "with_payload": True
        }
        
        response = requests.post(search_url, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Lỗi tìm kiếm: {response.status_code} - {response.text}")
    
    def format_search_results(self, query: str, results: Dict) -> str:
        """Format kết quả tìm kiếm để hiển thị đẹp"""
        output = []
        output.append(f"🔍 TRA CỨU: '{query}'")
        output.append("=" * 80)
        
        if 'result' in results and results['result']:
            for i, result in enumerate(results['result'], 1):
                score = result['score']
                text = result['payload']['text']
                metadata = result['payload']['metadata']
                
                output.append(f"\n📄 KẾT QUẢ {i} - Độ phù hợp: {score:.4f}")
                output.append(f"📚 Nguồn: {metadata['source']}")
                output.append(f"📝 Nội dung:")
                output.append(f"   {text}")
                output.append("-" * 80)
        else:
            output.append("\n❌ Không tìm thấy kết quả phù hợp")
        
        return "\n".join(output)
    
    async def get_collection_info(self) -> Dict:
        """Lấy thông tin collection"""
        info_url = f"{self.qdrant_url}/collections/{self.collection_name}"
        response = requests.get(info_url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Lỗi lấy thông tin collection: {response.status_code} - {response.text}")


async def main():
    """Hàm chính để test tra cứu luật kế toán"""
    searcher = VietnameseLawSearcher()
    
    print("🚀 BẮT ĐẦU TRA CỨU LUẬT KẾ TOÁN VIỆT NAM")
    print("=" * 80)
    
    # Kiểm tra thông tin collection
    try:
        collection_info = await searcher.get_collection_info()
        result = collection_info.get('result', {})
        points_count = result.get('points_count', 0)
        print(f"📊 Collection: {searcher.collection_name}")
        print(f"📈 Số điểm dữ liệu: {points_count}")
        print()
    except Exception as e:
        print(f"❌ Lỗi kết nối database: {e}")
        return
    
    # Danh sách câu hỏi tra cứu
    queries = [
        "kế toán trưởng có quyền và trách nhiệm gì",
        "điều kiện để làm kế toán viên",
        "báo cáo tài chính phải tuân thủ quy định nào",
        "doanh nghiệp kinh doanh dịch vụ kế toán cần điều kiện gì",
        "người làm kế toán có những quyền hạn gì",
        "chứng từ kế toán là gì",
        "đơn vị kế toán phải làm gì",
        "vi phạm luật kế toán bị xử lý như thế nào"
    ]
    
    # Thực hiện tra cứu từng câu hỏi
    for i, query in enumerate(queries, 1):
        try:
            print(f"\n🔎 TRUY VẤN {i}/{len(queries)}")
            results = await searcher.search_law(query, limit=2)
            formatted_results = searcher.format_search_results(query, results)
            print(formatted_results)
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"❌ Lỗi tra cứu '{query}': {e}")
    
    print("\n🎉 HOÀN THÀNH TRA CỨU LUẬT KẾ TOÁN!")


if __name__ == "__main__":
    asyncio.run(main())
