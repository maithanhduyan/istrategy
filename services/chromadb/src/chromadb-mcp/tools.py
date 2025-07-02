
import mcp.types as types
from mcp.server.lowlevel import Server
import chromadb
from chromadb.config import Settings
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.api import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    CohereEmbeddingFunction,
    OpenAIEmbeddingFunction,
    JinaEmbeddingFunction,
    VoyageAIEmbeddingFunction,
    RoboflowEmbeddingFunction,
)
import json
import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import os


# Global variables
_chroma_client = None
logger = logging.getLogger(__name__)

# Known embedding functions mapping
mcp_known_embedding_functions: Dict[str, EmbeddingFunction] = {
    "default": DefaultEmbeddingFunction,
    "cohere": CohereEmbeddingFunction,
    "openai": OpenAIEmbeddingFunction,
    "jina": JinaEmbeddingFunction,
    "voyageai": VoyageAIEmbeddingFunction,
    "roboflow": RoboflowEmbeddingFunction,
}

def get_chroma_client():
    """Get or create ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        try:
            # Try persistent client first
            data_dir = os.getenv('CHROMA_DATA_DIR', './chromadb')
            _chroma_client = chromadb.PersistentClient(path=data_dir)
            logger.info(f"ChromaDB persistent client initialized at {data_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize persistent client: {e}")
            # Fallback to ephemeral client
            _chroma_client = chromadb.EphemeralClient()
            logger.info("Using ephemeral ChromaDB client as fallback")
    return _chroma_client

def register_tools(server: Server):
    """Register all tools with the MCP server."""
    # This function is kept for compatibility but tools are handled in main.py
    pass

def get_tool_definitions() -> list[types.Tool]:
    """Get all tool definitions for registration."""
    return [
        # Echo tool
        types.Tool(
            name="echo",
            description="Echo back the provided message.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }
        ),
        
        # Collection Tools
        types.Tool(
            name="chroma_list_collections",
            description="List all collection names in the Chroma database with pagination support.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of collections to return"
                    },
                    "offset": {
                        "type": "integer", 
                        "description": "Optional number of collections to skip before returning results"
                    }
                }
            }
        ),
        
        types.Tool(
            name="chroma_create_collection",
            description="Create a new Chroma collection with configurable embedding function.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to create"
                    },
                    "embedding_function_name": {
                        "type": "string",
                        "description": "Name of the embedding function to use. Options: 'default', 'cohere', 'openai', 'jina', 'voyageai', 'roboflow'",
                        "default": "default"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata dict to add to the collection"
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        types.Tool(
            name="chroma_peek_collection",
            description="Peek at documents in a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to peek into"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of documents to peek at",
                        "default": 5
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        types.Tool(
            name="chroma_get_collection_info",
            description="Get information about a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to get info about"
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        types.Tool(
            name="chroma_get_collection_count",
            description="Get the number of documents in a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to count"
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        types.Tool(
            name="chroma_modify_collection",
            description="Modify a Chroma collection's name or metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to modify"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "Optional new name for the collection"
                    },
                    "new_metadata": {
                        "type": "object",
                        "description": "Optional new metadata for the collection"
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        types.Tool(
            name="chroma_delete_collection",
            description="Delete a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to delete"
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        # Document Tools
        types.Tool(
            name="chroma_add_documents",
            description="Add documents to a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to add documents to"
                    },
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text documents to add"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of IDs for the documents (required)"
                    },
                    "metadatas": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional list of metadata dictionaries for each document"
                    }
                },
                "required": ["collection_name", "documents", "ids"]
            }
        ),
        
        types.Tool(
            name="chroma_query_documents",
            description="Query documents from a Chroma collection with advanced filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to query"
                    },
                    "query_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of query texts to search for"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return per query",
                        "default": 5
                    },
                    "where": {
                        "type": "object",
                        "description": "Optional metadata filters using Chroma's query operators"
                    },
                    "where_document": {
                        "type": "object",
                        "description": "Optional document content filters"
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of what to include in response",
                        "default": ["documents", "metadatas", "distances"]
                    }
                },
                "required": ["collection_name", "query_texts"]
            }
        ),
        
        types.Tool(
            name="chroma_get_documents",
            description="Get documents from a Chroma collection with optional filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to get documents from"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of document IDs to retrieve"
                    },
                    "where": {
                        "type": "object",
                        "description": "Optional metadata filters using Chroma's query operators"
                    },
                    "where_document": {
                        "type": "object",
                        "description": "Optional document content filters"
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of what to include in response",
                        "default": ["documents", "metadatas"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of documents to return"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Optional number of documents to skip before returning results"
                    }
                },
                "required": ["collection_name"]
            }
        ),
        
        types.Tool(
            name="chroma_update_documents",
            description="Update documents in a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to update documents in"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to update (required)"
                    },
                    "embeddings": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "Optional list of new embeddings for the documents"
                    },
                    "metadatas": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional list of new metadata dictionaries for the documents"
                    },
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of new text documents"
                    }
                },
                "required": ["collection_name", "ids"]
            }
        ),
        
        types.Tool(
            name="chroma_delete_documents",
            description="Delete documents from a Chroma collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to delete documents from"
                    },
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to delete"
                    }
                },
                "required": ["collection_name", "ids"]
            }
        )
    ]