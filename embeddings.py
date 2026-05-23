"""
Embedding model wrapper using sentence-transformers (free, runs locally).
"""

import logging
from typing import List, Optional
import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wraps sentence-transformers for free, local embedding generation.
    Model: all-MiniLM-L6-v2 (fast, 384-dim, state-of-the-art for retrieval).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or CONFIG.embedding.model_name
        self._model = None
        self._langchain_embeddings = None
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

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings and return numpy array."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=CONFIG.embedding.normalize_embeddings,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()