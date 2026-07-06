"""
Text chunking utilities using LangChain's text splitters.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

from config import CONFIG

logger = logging.getLogger(__name__)


class SmartTextChunker:
    """
    Intelligent text chunker that splits documents into overlapping chunks.
    Supports multiple splitting strategies.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size or CONFIG.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or CONFIG.chunking.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size - 1
        self.strategy = strategy
        self._splitter = self._build_splitter()

    def _build_splitter(self):
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=CONFIG.chunking.separators,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.strategy == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n",
                length_function=len,
            )
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def chunk_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split a list of document dicts into smaller chunks."""
        chunks = []
        for doc in documents:
            text = doc.get("page_content", "")
            meta = doc.get("metadata", {})

            if not text.strip():
                continue

            split_texts = self._splitter.split_text(text)
            for idx, chunk_text in enumerate(split_texts):
                if chunk_text.strip():
                    chunks.append({
                        "page_content": chunk_text.strip(),
                        "metadata": {
                            **meta,
                            "chunk_index": idx,
                            "total_chunks": len(split_texts),
                        },
                    })

        logger.info(
            f"Chunked {len(documents)} document(s) into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap}, strategy={self.strategy})."
        )
        return chunks

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Convenience wrapper to chunk a plain string."""
        return self.chunk_documents([{"page_content": text, "metadata": metadata or {}}])

    def update_settings(self, chunk_size: int, chunk_overlap: int, strategy: str = "recursive"):
        """Hot-reload chunking settings."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size - 1
        self.strategy = strategy
        self._splitter = self._build_splitter()
        logger.info(f"Chunker updated: size={self.chunk_size}, overlap={self.chunk_overlap}, strategy={strategy}")