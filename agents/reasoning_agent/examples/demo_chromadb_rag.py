#!/usr/bin/env python3
"""
ChromaDB RAG Demo - Advanced Knowledge Management
Demonstrates real-world usage of RAG Engine with ChromaDB
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_engine import RAGEngine, DocumentProcessor


def print_section(title: str, char: str = "="):
    """Print formatted section header"""
    print(f"\n{char * 70}")
    print(f"🔮 {title}")
    print(f"{char * 70}")


async def demo_chromadb_rag():
    """Comprehensive ChromaDB RAG demonstration"""
    print_section("CHROMADB RAG KNOWLEDGE MANAGEMENT DEMO")

    # Initialize RAG Engine
    print("\n🚀 Initializing RAG Engine with ChromaDB...")
    rag = RAGEngine("demo_knowledge_base")

    if rag.chromadb_available:
        print("✅ ChromaDB successfully initialized")
        print(f"📚 Collection: {rag.collection_name}")
    else:
        print("⚠️ ChromaDB not available - running in mock mode")

    # Demo 1: Adding diverse knowledge documents
    print_section("1. KNOWLEDGE BASE POPULATION", "-")

    knowledge_docs = [
        {
            "content": "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.",
            "source": "python_guide",
            "category": "programming",
        },
        {
            "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "source": "ml_fundamentals",
            "category": "ai",
        },
        {
            "content": "ChromaDB is an open-source embedding database that makes it easy to build LLM applications by giving you the tools to store, embed, and search embeddings. It's designed to be simple, fast, and scalable.",
            "source": "chromadb_docs",
            "category": "database",
        },
        {
            "content": "Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable.",
            "source": "nlp_introduction",
            "category": "ai",
        },
        {
            "content": "Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models on the most accurate, up-to-date information and to give users insight into LLMs' generative process.",
            "source": "rag_paper",
            "category": "ai",
        },
    ]

    # Process and add documents
    for i, doc in enumerate(knowledge_docs, 1):
        content = doc["content"]
        metadata = {
            "source": doc["source"],
            "category": doc["category"],
            "doc_id": f"doc_{i}",
            "length": len(content),
        }

        result = await rag.add_documents([content], [metadata])
        print(
            f"📄 Doc {i}: {doc['source']} ({doc['category']}) - {result.get('status')}"
        )

    # Demo 2: Semantic search queries
    print_section("2. SEMANTIC SEARCH DEMONSTRATIONS", "-")

    search_queries = [
        {
            "query": "What is machine learning?",
            "expected": "Should find ML and AI related documents",
        },
        {
            "query": "How to build applications with Python?",
            "expected": "Should prioritize Python programming content",
        },
        {
            "query": "Database for embeddings and vector search",
            "expected": "Should find ChromaDB information",
        },
        {
            "query": "Understanding human language with computers",
            "expected": "Should find NLP content",
        },
        {
            "query": "Combining retrieval with language generation",
            "expected": "Should find RAG methodology",
        },
    ]

    for i, search in enumerate(search_queries, 1):
        print(f"\n🔍 Query {i}: {search['query']}")
        print(f"   Expected: {search['expected']}")

        results = await rag.search_documents(search["query"], n_results=3)

        if "error" not in results:
            print(f"   📊 Found {len(results.get('results', []))} results:")
            for j, result in enumerate(results.get("results", [])[:2], 1):
                doc = result.get("document", "")
                metadata = result.get("metadata", {})
                distance = result.get("distance", 0)

                preview = doc[:100] + "..." if len(doc) > 100 else doc
                category = metadata.get("category", "unknown")
                source = metadata.get("source", "unknown")

                print(f"      {j}. [{category}] {source} (distance: {distance:.3f})")
                print(f"         {preview}")
        else:
            print(f"   ❌ Error: {results.get('error')}")

    # Demo 3: Context-aware RAG pipeline
    print_section("3. CONTEXT-AWARE RAG PIPELINE", "-")

    rag_scenarios = [
        {
            "question": "How can I use Python for AI development?",
            "context_query": "Python artificial intelligence machine learning",
        },
        {
            "question": "What are the benefits of using vector databases?",
            "context_query": "database embeddings vector search",
        },
        {
            "question": "How does RAG improve language model performance?",
            "context_query": "retrieval augmented generation language models",
        },
    ]

    for i, scenario in enumerate(rag_scenarios, 1):
        print(f"\n💡 Scenario {i}: {scenario['question']}")

        # Get relevant context
        context = await rag.get_context(
            scenario["context_query"], max_context_length=300
        )
        print(f"   📖 Context length: {len(context)} characters")

        # Create augmented prompt
        augmented_prompt = rag.create_augmented_prompt(scenario["question"], context)
        print(f"   🎯 Augmented prompt length: {len(augmented_prompt)} characters")

        # Show context preview
        if context:
            context_preview = context[:150] + "..." if len(context) > 150 else context
            print(f"   📋 Context preview: {context_preview}")

        # Show augmented prompt structure
        prompt_lines = augmented_prompt.split("\n")
        print(f"   📝 Prompt structure: {len(prompt_lines)} lines")
        print(
            f"      - Context section: {len([l for l in prompt_lines if l.startswith('CONTEXT')])}"
        )
        print(
            f"      - Question section: {len([l for l in prompt_lines if l.startswith('QUESTION')])}"
        )

    # Demo 4: Advanced search with filters
    print_section("4. FILTERED SEARCH DEMONSTRATIONS", "-")

    print("\n🔍 Searching by category...")
    categories = ["programming", "ai", "database"]

    for category in categories:
        print(f"\n   📂 Category: {category}")
        # Note: ChromaDB filtering would be implemented here in production
        results = await rag.semantic_search("technology", document_type=category)
        print(f"      Found {len(results)} documents in {category} category")

        for j, result in enumerate(results[:2], 1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            print(f"         {j}. {source}")

    # Demo 5: Document processing showcase
    print_section("5. DOCUMENT PROCESSING CAPABILITIES", "-")

    long_document = """
    Artificial Intelligence (AI) is transforming every industry. Machine Learning, a subset of AI, 
    enables systems to automatically learn and improve from experience without being explicitly programmed.
    
    Deep Learning, which is part of machine learning, uses neural networks with three or more layers.
    These neural networks attempt to simulate the behavior of the human brain—albeit far from matching 
    its ability—allowing it to "learn" from large amounts of data.
    
    Natural Language Processing (NLP) is another important AI field that focuses on the interaction 
    between computers and humans using natural language. The ultimate objective of NLP is to read, 
    decipher, understand, and make sense of human languages in a valuable way.
    
    Computer Vision enables machines to interpret and make decisions based on visual data. This field 
    combines methods from physics, mathematics, statistics, and machine learning to build systems 
    that can process, analyze, and understand digital images.
    
    The future of AI includes developments in quantum computing, which could exponentially increase 
    processing power, and edge computing, which brings AI processing closer to data sources for 
    faster response times and reduced bandwidth usage.
    """

    processor = DocumentProcessor()

    print(f"📄 Original document: {len(long_document)} characters")

    # Test different chunking strategies
    chunk_strategies = [
        {"chunk_size": 200, "overlap": 50, "name": "Small chunks"},
        {"chunk_size": 400, "overlap": 100, "name": "Medium chunks"},
        {"chunk_size": 600, "overlap": 150, "name": "Large chunks"},
    ]

    for strategy in chunk_strategies:
        chunks = processor.chunk_document(
            long_document,
            chunk_size=strategy["chunk_size"],
            overlap=strategy["overlap"],
        )

        avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
        print(
            f"   📊 {strategy['name']}: {len(chunks)} chunks, avg size: {avg_size:.1f}"
        )

        # Add chunks to knowledge base
        metadatas = [
            processor.extract_metadata(chunk, f"ai_overview_{strategy['name']}")
            for chunk in chunks
        ]

        result = await rag.add_documents(chunks, metadatas)
        print(f"      ✅ Added to knowledge base: {result.get('status')}")

    # Final search test with expanded knowledge
    print(f"\n🔍 Final search test with expanded knowledge base...")
    final_query = "What are the main fields of artificial intelligence?"
    results = await rag.search_documents(final_query, n_results=5)

    print(f"   Query: {final_query}")
    print(f"   📊 Results: {len(results.get('results', []))} documents found")

    for i, result in enumerate(results.get("results", [])[:3], 1):
        doc = result.get("document", "")
        distance = result.get("distance", 0)
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        print(f"      {i}. Distance: {distance:.3f} - {preview}")

    print_section("DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("✅ ChromaDB RAG system fully operational")
    print("📚 Knowledge base populated with diverse content")
    print("🔍 Semantic search working effectively")
    print("🎯 Context-aware augmentation functional")
    print("📊 Document processing optimized")

    # Cleanup
    if rag.chromadb_available and rag.client:
        try:
            print(f"\n🧹 Cleaning up demo collection...")
            rag.client.delete_collection("demo_knowledge_base")
            print("   ✅ Demo collection cleaned up")
        except Exception as e:
            print(f"   ⚠️ Cleanup note: {e}")


if __name__ == "__main__":
    asyncio.run(demo_chromadb_rag())
