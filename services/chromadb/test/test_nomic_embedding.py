#!/usr/bin/env python3
"""
Test script for Nomic Embedding v2 MoE integration with ChromaDB.
"""

import sys
import os

# Add the chromadb-mcp directory to Python path
chromadb_mcp_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'chromadb-mcp')
sys.path.insert(0, os.path.abspath(chromadb_mcp_path))

import asyncio
import json
from embedding import (
    NomicEmbeddingFunction,
    NomicQueryEmbeddingFunction,
    NomicDocumentEmbeddingFunction,
    NomicClusteringEmbeddingFunction,
)


def test_nomic_embedding_basic():
    """Test basic Nomic embedding functionality."""
    print("🧪 Testing Nomic Embedding Basic Functionality...")
    
    try:
        # Test document embedding
        doc_embedding = NomicDocumentEmbeddingFunction()
        test_texts = [
            "Xin chào, tôi là một văn bản tiếng Việt về công nghệ AI.",
            "Hello, this is an English text about artificial intelligence.",
            "ChromaDB is a powerful vector database for AI applications."
        ]
        
        print(f"📝 Testing with {len(test_texts)} documents...")
        embeddings = doc_embedding(test_texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📊 Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
        
        # Test query embedding
        query_embedding = NomicQueryEmbeddingFunction()
        query_text = ["trí tuệ nhân tạo"]
        
        print(f"🔍 Testing query embedding...")
        query_emb = query_embedding(query_text)
        
        print(f"✅ Generated query embedding with dimension: {len(query_emb[0]) if query_emb else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic test: {str(e)}")
        return False


def test_nomic_embedding_config():
    """Test Nomic embedding configuration."""
    print("\n🔧 Testing Nomic Embedding Configuration...")
    
    try:
        # Test build from config
        config = {
            "prompt_name": "query"
        }
        
        embedding_func = NomicEmbeddingFunction.build_from_config(config)
        print(f"✅ Built from config: {embedding_func.get_config()}")
        
        # Test validation
        NomicEmbeddingFunction.validate_config(config)
        print("✅ Config validation passed")
        
        # Test invalid config
        try:
            invalid_config = {"prompt_name": "invalid_prompt"}
            NomicEmbeddingFunction.validate_config(invalid_config)
            print("❌ Should have failed validation")
            return False
        except ValueError:
            print("✅ Invalid config correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in config test: {str(e)}")
        return False


def test_chromadb_integration():
    """Test integration with ChromaDB."""
    print("\n🗄️ Testing ChromaDB Integration...")
    
    try:
        import chromadb
        from chromadb.api.collection_configuration import CreateCollectionConfiguration
        
        # Create ephemeral client for testing
        client = chromadb.EphemeralClient()
        
        # Test with Nomic document embedding
        doc_embedding = NomicDocumentEmbeddingFunction()
        
        configuration = CreateCollectionConfiguration(
            embedding_function=doc_embedding
        )
        
        collection = client.create_collection(
            name="test_nomic_collection",
            configuration=configuration
        )
        
        print("✅ Created ChromaDB collection with Nomic embedding")
        
        # Add Vietnamese documents
        vietnamese_docs = [
            "Trí tuệ nhân tạo đang phát triển rất nhanh trong những năm gần đây.",
            "ChromaDB là một cơ sở dữ liệu vector mạnh mẽ cho các ứng dụng AI.",
            "Machine learning và deep learning là những công nghệ quan trọng của AI."
        ]
        
        collection.add(
            documents=vietnamese_docs,
            ids=["doc1", "doc2", "doc3"],
            metadatas=[
                {"language": "vietnamese", "topic": "AI"},
                {"language": "vietnamese", "topic": "database"},
                {"language": "vietnamese", "topic": "ML"}
            ]
        )
        
        print(f"✅ Added {len(vietnamese_docs)} Vietnamese documents")
        
        # Test semantic search
        query_results = collection.query(
            query_texts=["công nghệ AI"],
            n_results=2
        )
        
        print(f"✅ Query results: {len(query_results['documents'][0])} relevant documents found")
        for i, doc in enumerate(query_results['documents'][0]):
            distance = query_results['distances'][0][i]
            print(f"   📄 Document {i+1} (distance: {distance:.4f}): {doc[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in ChromaDB integration test: {str(e)}")
        return False


def test_performance():
    """Test performance with larger dataset."""
    print("\n⚡ Testing Performance...")
    
    try:
        embedding_func = NomicDocumentEmbeddingFunction()
        
        # Generate test documents
        test_docs = [
            f"This is test document number {i} about artificial intelligence and machine learning."
            for i in range(50)
        ]
        
        import time
        start_time = time.time()
        
        embeddings = embedding_func(test_docs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Generated {len(embeddings)} embeddings in {duration:.2f} seconds")
        print(f"📊 Average: {duration/len(embeddings)*1000:.2f} ms per document")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in performance test: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Nomic Embedding v2 MoE Tests for ChromaDB Integration\n")
    
    tests = [
        ("Basic Functionality", test_nomic_embedding_basic),
        ("Configuration", test_nomic_embedding_config),
        ("ChromaDB Integration", test_chromadb_integration),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {str(e)}\n")
            results.append((test_name, False))
    
    # Summary
    print(f"{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Nomic Embedding v2 MoE is ready for ChromaDB!")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
