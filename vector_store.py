"""
Vector store manager supporting FAISS (default) and Chroma.
Includes hybrid search (dense + BM25 sparse).
"""

import logging
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from config import CONFIG
from embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector storage and retrieval.
    Supports:
      - Dense search (FAISS)
      - Sparse search (BM25 via rank_bm25)
      - Hybrid search (Reciprocal Rank Fusion)
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self._faiss_index = None
        self._chroma_db = None
        self._documents: List[Dict[str, Any]] = []
        self._bm25 = None
        self._store_type = CONFIG.vector_store.store_type

    # ------------------------------------------------------------------ #
    #  Build / Reset                                                       #
    # ------------------------------------------------------------------ #

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Index a list of chunk dicts into the vector store."""
        if not chunks:
            raise ValueError("No chunks provided to index.")

        self._documents = chunks
        texts = [c["page_content"] for c in chunks]

        if self._store_type == "faiss":
            self._build_faiss(texts)
        else:
            self._build_chroma(chunks)

        self._build_bm25(texts)
        logger.info(f"Vector store built with {len(chunks)} chunks.")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Incrementally add chunks to an existing store."""
        if not chunks:
            return

        # If no store exists yet, just do a full build.
        if not self._documents:
            self.build_from_chunks(chunks)
            return

        new_texts = [c["page_content"] for c in chunks]
        new_metadatas = [c["metadata"] for c in chunks]

        # --- Dense index: add only new vectors ---
        if self._store_type == "faiss" and self._faiss_index:
            self._faiss_index.add_texts(new_texts, metadatas=new_metadatas)
        elif self._store_type == "chroma" and self._chroma_db:
            self._chroma_db.add_texts(new_texts, metadatas=new_metadatas)

        # --- Append to document list ---
        self._documents.extend(chunks)

        # --- Rebuild BM25 over all documents (cheap, no embeddings) ---
        all_texts = [c["page_content"] for c in self._documents]
        self._build_bm25(all_texts)

        logger.info(
            f"Incrementally added {len(chunks)} chunks "
            f"(total: {len(self._documents)})."
        )

    def _build_faiss(self, texts: List[str]) -> None:
        import faiss
        lc_embeddings = self.embedding_model.get_langchain_embeddings()
        from langchain_community.vectorstores import FAISS
        self._faiss_index = FAISS.from_texts(
            texts,
            lc_embeddings,
            metadatas=[c["metadata"] for c in self._documents],
        )

    def _build_chroma(self, chunks: List[Dict[str, Any]]) -> None:
        from langchain_community.vectorstores import Chroma
        lc_embeddings = self.embedding_model.get_langchain_embeddings()
        self._chroma_db = Chroma.from_texts(
            [c["page_content"] for c in chunks],
            lc_embeddings,
            metadatas=[c["metadata"] for c in chunks],
            persist_directory=CONFIG.vector_store.persist_directory,
        )

    def _build_bm25(self, texts: List[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 disabled.")
            self._bm25 = None

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def dense_search(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Pure dense (embedding-based) similarity search."""
        k = top_k or CONFIG.vector_store.top_k
        if self._store_type == "faiss" and self._faiss_index:
            results = self._faiss_index.similarity_search_with_score(query, k=k)
            return [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "search_type": "dense",
                }
                for doc, score in results
            ]
        elif self._store_type == "chroma" and self._chroma_db:
            results = self._chroma_db.similarity_search_with_score(query, k=k)
            return [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "search_type": "dense",
                }
                for doc, score in results
            ]
        return []

    def sparse_search(
        self, query: str, top_k: int = None
    ) -> List[Dict[str, Any]]:
        """BM25 keyword-based sparse search."""
        if self._bm25 is None or not self._documents:
            return []
        k = top_k or CONFIG.vector_store.top_k
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [
            {
                "page_content": self._documents[i]["page_content"],
                "metadata": self._documents[i]["metadata"],
                "score": float(scores[i]),
                "search_type": "sparse",
            }
            for i in ranked_indices
            if scores[i] > 0
        ]

    def hybrid_search(
        self, query: str, top_k: int = None, alpha: float = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: Reciprocal Rank Fusion of dense + sparse results.
        alpha=1.0 → pure dense; alpha=0.0 → pure sparse.
        """
        k = top_k or CONFIG.vector_store.top_k
        alpha = alpha if alpha is not None else CONFIG.vector_store.hybrid_alpha

        dense = self.dense_search(query, top_k=k * 2)
        sparse = self.sparse_search(query, top_k=k * 2)

        rrf_scores: Dict[str, float] = {}
        content_map: Dict[str, Dict] = {}

        def add_rrf(results, weight, rrf_k=60):
            for rank, r in enumerate(results):
                key = r["page_content"][:200]
                content_map[key] = r
                rrf_scores[key] = rrf_scores.get(key, 0) + weight * (1 / (rrf_k + rank + 1))

        add_rrf(dense, alpha)
        add_rrf(sparse, 1 - alpha)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            {**content_map[key], "rrf_score": score, "search_type": "hybrid"}
            for key, score in ranked
        ]

    def retrieve(
        self, query: str, top_k: int = None, mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Main retrieval entry point."""
        if not self._documents:
            return []
        if mode == "dense":
            return self.dense_search(query, top_k)
        elif mode == "sparse":
            return self.sparse_search(query, top_k)
        else:
            return self.hybrid_search(query, top_k)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str = "vector_store_cache.pkl") -> None:
        data = {
            "documents": self._documents,
            "store_type": self._store_type,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
        if self._store_type == "faiss" and self._faiss_index:
            self._faiss_index.save_local("faiss_index")
            
        logger.info(f"Vector store saved to {path} and faiss_index directory.")

    def load(self, path: str = "vector_store_cache.pkl") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._store_type = data["store_type"]
        texts = [d["page_content"] for d in self._documents]
        
        if self._store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            lc_emb = self.embedding_model.get_langchain_embeddings()
            
            # Support legacy byte-serialized format
            if "faiss_bytes" in data:
                self._faiss_index = FAISS.deserialize_from_bytes(
                    data["faiss_bytes"], lc_emb
                )
            # Load from standard faiss_index directory
            elif os.path.exists("faiss_index"):
                try:
                    self._faiss_index = FAISS.load_local("faiss_index", lc_emb, allow_dangerous_deserialization=True)
                except TypeError:
                    # Fallback for older langchain versions
                    self._faiss_index = FAISS.load_local("faiss_index", lc_emb)
                    
        elif self._store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            lc_emb = self.embedding_model.get_langchain_embeddings()
            self._chroma_db = Chroma(
                persist_directory=CONFIG.vector_store.persist_directory,
                embedding_function=lc_emb
            )
                    
        self._build_bm25(texts)
        logger.info(f"Vector store loaded from {path} ({len(self._documents)} chunks).")
        return True

    @property
    def is_ready(self) -> bool:
        if not self._documents:
            return False
        if self._store_type == "faiss":
            return self._faiss_index is not None
        if self._store_type == "chroma":
            return self._chroma_db is not None
        return False

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._faiss_index = None
        self._chroma_db = None
        self._documents = []
        self._bm25 = None
        logger.info("Vector store cleared.")

    def get_document_sources(self) -> List[str]:
        sources = set()
        for d in self._documents:
            src = d.get("metadata", {}).get("source", "Unknown")
            sources.add(src)
        return sorted(sources)