"""
Embedding model wrapper using sentence-transformers (free, runs locally).
"""

import hashlib
import logging
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)

# Maximum number of unique text strings kept in the per-instance embedding cache.
# 512 entries × 384 float32 values × 4 bytes ≈ 786 KB — well within reason.
_EMBED_CACHE_MAX_SIZE: int = 512


class EmbeddingModel:
    """
    Wraps sentence-transformers for free, local embedding generation.
    Model: all-MiniLM-L6-v2 (fast, 384-dim, state-of-the-art for retrieval).

    Embedding calls are served from a per-instance SHA-256-keyed LRU cache
    (OrderedDict, max 512 entries) so that identical strings — such as a query
    that gets encoded once explicitly and a second time internally by the
    LangChain FAISS wrapper — never hit the model more than once.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CONFIG.embedding.model_name
        self._model = None
        self._langchain_embeddings = None
        # Per-instance LRU cache: sha256(text) -> np.ndarray (1-D vector)
        self._embed_cache: OrderedDict = OrderedDict()
        logger.info(f"Embedding model configured: {self.model_name}")

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Sentence-transformers model loaded.")

    def get_langchain_embeddings(self):
        """Return a LangChain-compatible embedding object."""
        if self._langchain_embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._langchain_embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": CONFIG.embedding.device},
                encode_kwargs={"normalize_embeddings": CONFIG.embedding.normalize_embeddings},
            )
        return self._langchain_embeddings

    def _cache_key(self, text: str) -> str:
        """Return a compact, collision-resistant key for *text*."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings and return a numpy array.

        Results are served from a per-instance LRU cache (max 512 unique
        strings). Only the uncached subset is forwarded to the model, then
        the vectors are merged back in the original order.
        """
        self._load_model()

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # --- Pass 1: resolve cache hits ---
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._embed_cache:
                self._embed_cache.move_to_end(key)   # mark as recently used
                results[i] = self._embed_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # --- Pass 2: encode only the missing texts in one batched call ---
        if uncached_texts:
            new_vecs = self._model.encode(
                uncached_texts,
                normalize_embeddings=CONFIG.embedding.normalize_embeddings,
                show_progress_bar=False,
                batch_size=32,
            )
            for j, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                vec = new_vecs[j]
                results[idx] = vec
                key = self._cache_key(text)
                self._embed_cache[key] = vec
                # Evict the least-recently-used entry when the cap is exceeded
                if len(self._embed_cache) > _EMBED_CACHE_MAX_SIZE:
                    self._embed_cache.popitem(last=False)

        hits = len(texts) - len(uncached_texts)
        logger.debug(
            "Embedding cache — hits: %d, misses: %d, cache size: %d/%d",
            hits, len(uncached_texts), len(self._embed_cache), _EMBED_CACHE_MAX_SIZE,
        )
        return np.array(results)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string (cache-aware via embed_texts)."""
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()