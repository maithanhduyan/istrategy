# -*- coding: utf-8 -*-
import asyncio
from app.chromadb import chroma_create_collection, chroma_list_collections

async def test():
    try:
        # Tạo collection test
        result = await chroma_create_collection('test_collection', metadata={'description': 'Test collection for MCP'})
        print('Created collection:', result)
        
        # List collections
        collections = await chroma_list_collections()
        print('Collections:', collections)
    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    asyncio.run(test())
