"""
Central configuration for the RAG Chatbot.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    store_type: str = "faiss"          # "faiss" or "chroma"
    persist_directory: str = "./vector_store"
    top_k: int = 5
    score_threshold: float = 0.3
    hybrid_alpha: float = 0.5          # blend weight: 1.0 = dense only, 0.0 = sparse only


@dataclass
class LLMConfig:
    provider: str = "groq"             # "groq" | "huggingface" | "together"
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 1024
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    huggingface_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY")
    )


@dataclass
class ConversationConfig:
    max_history_turns: int = 10        # pairs kept in memory
    use_chat_history: bool = True


@dataclass
class ScrapingConfig:
    max_pages: int = 5
    timeout: int = 15
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class EvaluationConfig:
    enabled: bool = True
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    retrieval_precision_threshold: float = 0.6


@dataclass
class RAGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Supported languages for multilingual mode
    supported_languages: List[str] = field(
        default_factory=lambda: [
            "en", "es", "fr", "de", "it", "pt", "nl",
            "hi", "zh", "ja", "ko", "ar", "ru"
        ]
    )


# Singleton instance
CONFIG = RAGConfig()