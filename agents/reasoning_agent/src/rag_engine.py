"""
RAG Engine for Reasoning Agent
Provides Retrieval-Augmented Generation capabilities using ChromaDB
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available, using mock implementation")


class RAGEngine:
    """RAG implementation using ChromaDB for document retrieval and context augmentation"""

    def __init__(self, collection_name: str = "reasoning_agent_knowledge"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.chromadb_available = CHROMADB_AVAILABLE
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        if not self.chromadb_available:
            print("ChromaDB not available, using mock implementation")
            return

        try:
            # Initialize ChromaDB client with explicit persist_directory
            persist_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
            )
            self.client = chromadb.Client(Settings(persist_directory=persist_dir))

            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(
                    f"✅ Connected to existing ChromaDB collection: {self.collection_name} (persist: {persist_dir})"
                )
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name
                )
                print(
                    f"✅ Created new ChromaDB collection: {self.collection_name} (persist: {persist_dir})"
                )

        except Exception as e:
            print(f"⚠️ ChromaDB initialization failed: {e}")
            self.chromadb_available = False

    def _check_mcp_availability(self) -> bool:
        """Check if MCP ChromaDB is available"""
        return self.chromadb_available

    async def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add documents to the knowledge base"""
        if not self.chromadb_available or not self.collection:
            return {"error": "ChromaDB not available", "status": "mock"}

        if not ids:
            ids = [
                f"doc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                for i in range(len(documents))
            ]

        if not metadatas:
            metadatas = [
                {"timestamp": datetime.now().isoformat(), "source": "reasoning_agent"}
                for _ in documents
            ]

        try:
            # Add documents to ChromaDB
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

            result = {
                "status": "success",
                "documents_added": len(documents),
                "collection": self.collection_name,
                "ids": ids,
            }
            print(f"✅ Added {len(documents)} documents to ChromaDB collection")

        except Exception as e:
            result = {"status": "error", "error": str(e), "documents_added": 0}
            print(f"❌ Error adding documents: {e}")

        return result

    async def search_documents(
        self, query: str, n_results: int = 5, where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Search for relevant documents based on query"""
        if not self.chromadb_available or not self.collection:
            # Return mock data when ChromaDB not available
            return {
                "query": query,
                "results": [
                    {
                        "document": f"Sample document content relevant to: {query}",
                        "metadata": {"source": "knowledge_base", "relevance": 0.95},
                        "distance": 0.1,
                    }
                ],
                "n_results": n_results,
                "status": "mock",
            }

        try:
            # Query ChromaDB collection
            query_results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            results = []
            if query_results["documents"] and query_results["documents"][0]:
                for i, doc in enumerate(query_results["documents"][0]):
                    result_item = {
                        "document": doc,
                        "metadata": (
                            query_results["metadatas"][0][i]
                            if query_results["metadatas"]
                            else {}
                        ),
                        "distance": (
                            query_results["distances"][0][i]
                            if query_results["distances"]
                            else 0.0
                        ),
                    }
                    results.append(result_item)

            result = {
                "query": query,
                "results": results,
                "n_results": len(results),
                "status": "success",
            }
            print(f"🔍 Found {len(results)} documents for query: {query[:50]}...")

        except Exception as e:
            result = {
                "query": query,
                "error": str(e),
                "results": [],
                "n_results": 0,
                "status": "error",
            }
            print(f"❌ Error searching documents: {e}")

        return result

    async def get_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query to augment generation"""
        search_results = await self.search_documents(query)

        if "error" in search_results:
            return ""

        context_pieces = []
        current_length = 0

        for result in search_results.get("results", []):
            doc_content = result.get("document", "")
            if current_length + len(doc_content) <= max_context_length:
                context_pieces.append(doc_content)
                current_length += len(doc_content)
            else:
                # Truncate the last piece to fit
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if substantial space left
                    context_pieces.append(doc_content[:remaining_space] + "...")
                break

        return "\n\n".join(context_pieces)

    async def semantic_search(
        self, query: str, document_type: Optional[str] = None
    ) -> List[Dict]:
        """Perform semantic search with optional filtering"""
        where_clause = {"document_type": document_type} if document_type else None
        results = await self.search_documents(query, where=where_clause)
        return results.get("results", [])

    def create_augmented_prompt(self, original_query: str, context: str) -> str:
        """Create an augmented prompt with retrieved context"""
        if not context.strip():
            return original_query

        augmented_prompt = f"""Based on the following relevant context, please answer the question:

CONTEXT:
{context}

QUESTION: {original_query}

Please provide a comprehensive answer using the context above when relevant, and indicate if you're drawing from the provided context or your general knowledge."""

        return augmented_prompt


class DocumentProcessor:
    """Process and prepare documents for RAG system"""

    @staticmethod
    def chunk_document(
        text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[str]:
        """Split document into overlapping chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at word boundaries
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > chunk_size * 0.7:  # Don't break too early
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    @staticmethod
    def extract_metadata(text: str, source: str = "") -> Dict[str, Any]:
        """Extract metadata from document"""
        return {
            "source": source,
            "length": len(text),
            "word_count": len(text.split()),
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and preprocess text for better retrieval"""
        # Basic preprocessing
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        return text


# RAG Tool Functions for ToolExecutor integration
class RAGTools:
    """RAG tools for integration with ToolExecutor"""

    def __init__(self):
        self.rag_engine = RAGEngine()
        self.processor = DocumentProcessor()

    async def rag_add_knowledge(self, args: List[str]) -> str:
        """Add knowledge to RAG system: rag_add_knowledge(text, source)"""
        if len(args) < 1:
            return "Error: rag_add_knowledge requires at least 1 argument (text)"

        text = args[0]
        source = args[1] if len(args) > 1 else "user_input"

        try:
            # Process document
            chunks = self.processor.chunk_document(text)
            metadatas = [
                self.processor.extract_metadata(chunk, source) for chunk in chunks
            ]

            # Add to knowledge base
            result = await self.rag_engine.add_documents(chunks, metadatas)

            return f"Added {len(chunks)} chunks to knowledge base. Status: {result.get('status')}"

        except Exception as e:
            return f"Error adding knowledge: {str(e)}"

    async def rag_search(self, args: List[str]) -> str:
        """Search knowledge base: rag_search(query, max_results)"""
        if len(args) < 1:
            return "Error: rag_search requires 1 argument (query)"

        query = args[0]
        max_results = int(args[1]) if len(args) > 1 else 5

        try:
            results = await self.rag_engine.search_documents(query, max_results)

            if "error" in results:
                return f"Search error: {results['error']}"

            search_results = results.get("results", [])
            if not search_results:
                return "No relevant documents found"

            response = f"Found {len(search_results)} relevant documents:\n"
            for i, result in enumerate(search_results, 1):
                doc = (
                    result.get("document", "")[:200] + "..."
                    if len(result.get("document", "")) > 200
                    else result.get("document", "")
                )
                response += f"{i}. {doc}\n"

            return response

        except Exception as e:
            return f"Error searching: {str(e)}"

    async def rag_augmented_query(self, args: List[str]) -> str:
        """Create augmented query with context: rag_augmented_query(question)"""
        if len(args) < 1:
            return "Error: rag_augmented_query requires 1 argument (question)"

        question = args[0]

        try:
            context = await self.rag_engine.get_context(question)
            augmented_prompt = self.rag_engine.create_augmented_prompt(
                question, context
            )

            return f"Augmented prompt created with {len(context)} characters of context"

        except Exception as e:
            return f"Error creating augmented query: {str(e)}"


# Integration helper
def get_rag_tools() -> Dict[str, callable]:
    """Get RAG tools for ToolExecutor integration"""
    rag_tools = RAGTools()

    return {
        "rag_add_knowledge": rag_tools.rag_add_knowledge,
        "rag_search": rag_tools.rag_search,
        "rag_augmented_query": rag_tools.rag_augmented_query,
    }
