#!/usr/bin/env python3
"""
Test RAG Engine with Real ChromaDB Integration
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_engine import RAGEngine, DocumentProcessor


async def test_rag_chromadb():
    """Test RAG Engine with ChromaDB"""
    print("🧪 Testing RAG Engine with ChromaDB Integration")
    print("=" * 60)

    # Initialize RAG Engine
    print("\n1. Initializing RAG Engine...")
    rag = RAGEngine("test_reasoning_agent")

    if not rag.chromadb_available:
        print("⚠️ ChromaDB not available, running in mock mode")
    else:
        print("✅ ChromaDB available and initialized")

    # Test documents
    test_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Python is a versatile programming language widely used in data science and machine learning applications.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
        "Natural language processing enables computers to understand, interpret and generate human language.",
        "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers.",
    ]

    # Test adding documents
    print("\n2. Adding test documents...")
    result = await rag.add_documents(test_docs)
    print(f"   Status: {result.get('status')}")
    print(f"   Documents added: {result.get('documents_added', 0)}")

    # Test searching
    test_queries = [
        "What is machine learning?",
        "How is Python used in AI?",
        "Explain neural networks",
        "What is deep learning?",
    ]

    print("\n3. Testing document search...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        search_result = await rag.search_documents(query, n_results=3)

        if "error" in search_result:
            print(f"   ❌ Error: {search_result['error']}")
        else:
            print(f"   ✅ Found {len(search_result.get('results', []))} results")
            for j, result in enumerate(search_result.get("results", [])[:2], 1):
                doc_preview = (
                    result.get("document", "")[:100] + "..."
                    if len(result.get("document", "")) > 100
                    else result.get("document", "")
                )
                distance = result.get("distance", 0)
                print(f"      {j}. Distance: {distance:.3f} - {doc_preview}")

    # Test context retrieval
    print("\n4. Testing context retrieval...")
    test_context_query = "artificial intelligence and machine learning"
    context = await rag.get_context(test_context_query, max_context_length=500)
    print(f"   Query: {test_context_query}")
    print(f"   Context length: {len(context)} characters")
    if context:
        print(f"   Context preview: {context[:200]}...")

    # Test augmented prompt
    print("\n5. Testing augmented prompt creation...")
    original_query = "How can I use Python for AI projects?"
    augmented_prompt = rag.create_augmented_prompt(original_query, context)
    print(f"   Original query: {original_query}")
    print(f"   Augmented prompt length: {len(augmented_prompt)} characters")
    print(f"   Augmented prompt preview:\n   {augmented_prompt[:300]}...")

    # Test document processing
    print("\n6. Testing document processing...")
    processor = DocumentProcessor()

    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so.
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.
    """

    chunks = processor.chunk_document(long_text, chunk_size=200, overlap=50)
    print(f"   Original text length: {len(long_text)} characters")
    print(f"   Number of chunks: {len(chunks)}")
    print(
        f"   Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} characters"
    )

    metadata = processor.extract_metadata(long_text, source="test_document")
    print(f"   Extracted metadata: {metadata}")

    print("\n🎉 RAG Engine testing completed!")

    # Cleanup (optional)
    if rag.chromadb_available and rag.client:
        try:
            print("\n7. Cleaning up test collection...")
            rag.client.delete_collection("test_reasoning_agent")
            print("   ✅ Test collection deleted")
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    asyncio.run(test_rag_chromadb())
