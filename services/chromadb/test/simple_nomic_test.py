#!/usr/bin/env python3
"""
Simple test for Nomic embedding integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'chromadb-mcp'))

from embedding import NomicDocumentEmbeddingFunction, NomicQueryEmbeddingFunction

def main():
    print("🧪 Simple Nomic Embedding Test")
    print("=" * 50)
    
    try:
        # Test document embedding
        print("1. Testing document embedding...")
        doc_func = NomicDocumentEmbeddingFunction()
        doc_result = doc_func(["Xin chào, tôi là văn bản tiếng Việt về AI"])
        print(f"   ✅ Document embedding dimension: {len(doc_result[0])}")
        
        # Test query embedding  
        print("2. Testing query embedding...")
        query_func = NomicQueryEmbeddingFunction()
        query_result = query_func(["trí tuệ nhân tạo"])
        print(f"   ✅ Query embedding dimension: {len(query_result[0])}")
        
        print("\n🎉 All tests passed! Nomic embedding is working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
