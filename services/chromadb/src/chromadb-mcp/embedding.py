from typing import Dict, Any, List, Optional, cast
import logging
import asyncio

from chromadb.api.types import (
    EmbeddingFunction,
    Embeddings,
    Documents,
)

logger = logging.getLogger(__name__)

# Preload model at module level for performance
_nomic_model = None
_nomic_model_lock = asyncio.Lock()
_nomic_model_ready = asyncio.Event()

async def preload_nomic_model_async():
    global _nomic_model
    async with _nomic_model_lock:
        if _nomic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                loop = asyncio.get_event_loop()
                _nomic_model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
                )
                _nomic_model_ready.set()
            except ImportError:
                raise ImportError("sentence-transformers required: pip install sentence-transformers")
    return _nomic_model

# Synchronous fallback for legacy code

def preload_nomic_model():
    global _nomic_model
    if _nomic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _nomic_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")
    return _nomic_model

class NomicEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Nomic Embedding v2 MoE function for ChromaDB integration.
    Simple, lightweight implementation using sentence-transformers.
    """
    
    def __init__(self, prompt_name: str = "document") -> None:
        """Initialize with prompt type: 'document', 'query', 'Clustering', etc."""
        self.model_name = "nomic-ai/nomic-embed-text-v2-moe"
        self.prompt_name = prompt_name
        self._model = None

    def _get_model(self):
        """Get model with async preload support."""
        global _nomic_model
        if _nomic_model is not None:
            self._model = _nomic_model
        elif self._model is None:
            # Check if async preload is ready
            if _nomic_model_ready.is_set():
                self._model = _nomic_model
            else:
                # Fallback to sync preload if async not ready
                self._model = preload_nomic_model()
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for documents."""
        if not input:
            return []
        
        model = self._get_model()
        texts = input if isinstance(input, list) else [input]
        
        # Generate embeddings with prompt
        embeddings = model.encode(texts, prompt_name=self.prompt_name, convert_to_tensor=False)
        
        # Convert to lists for ChromaDB
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "NomicEmbeddingFunction":
        """Build from config dict."""
        prompt_name = config.get("prompt_name", "document") if config else "document"
        return NomicEmbeddingFunction(prompt_name=prompt_name)

    @staticmethod
    def name() -> str:
        return "nomic_embedding_v2_moe"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "prompt_name": self.prompt_name}

    def max_tokens(self) -> int:
        return 8192

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Simple validation."""
        if not config or "prompt_name" not in config:
            return
        
        valid_prompts = ["document", "query", "passage", "Clustering", "Classification", 
                        "MultilabelClassification", "PairClassification", "STS", 
                        "Summarization", "Speed"]
        
        prompt = config["prompt_name"]
        if prompt not in valid_prompts:
            raise ValueError(f"Invalid prompt_name '{prompt}'. Use: {valid_prompts}")

# Preload model at import for best performance (async)
async def _init_async_preload():
    """Initialize async preload in background."""
    try:
        await preload_nomic_model_async()
        logger.info("Nomic embedding model preloaded successfully (async)")
    except Exception as e:
        logger.warning(f"Async preload failed: {e}, falling back to sync")
        preload_nomic_model()

# Start async preload if possible
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If event loop is running, schedule the task
        asyncio.create_task(_init_async_preload())
    else:
        # If no event loop, start sync preload
        preload_nomic_model()
        logger.info("Nomic embedding model preloaded successfully (sync)")
except RuntimeError:
    # If not in event loop, fallback to sync preload
    preload_nomic_model()
    logger.info("Nomic embedding model preloaded successfully (sync fallback)")