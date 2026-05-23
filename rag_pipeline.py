"""
Core RAG Pipeline — orchestrates all components end-to-end.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import CONFIG
from document_loader import load_multiple_documents
from text_chunker import SmartTextChunker
from embeddings import EmbeddingModel
from vector_store import VectorStoreManager
from llm_provider import LLMProvider
from evaluation import RAGEvaluator, EvaluationResult
from multilingual import MultilingualHandler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Prompt Templates                                                    #
# ------------------------------------------------------------------ #

RAG_SYSTEM_PROMPT = """You are a knowledgeable and precise assistant. Answer the user's question
using ONLY the information provided in the context below. If the context does not contain enough
information to answer the question, say so clearly — do not hallucinate or make up facts.

Structure your answers clearly with:
- A direct answer to the question
- Supporting evidence from the context
- Source references when available

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:"""

STANDALONE_QUESTION_PROMPT = """Given the chat history below and a follow-up question,
rephrase the follow-up question to be a standalone question that captures all necessary context.

Chat History:
{history}

Follow-up Question: {question}

Standalone Question:"""


class ConversationMemory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or CONFIG.conversation.max_history_turns
        self._history: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self._history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        # Keep only the last max_turns pairs
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def format_history(self) -> str:
        if not self._history:
            return "No previous conversation."
        lines = []
        for msg in self._history[-6:]:  # last 3 turns for context
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self._history = []

    def __len__(self):
        return len(self._history)


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    Handles: document ingestion → chunking → embedding → retrieval → LLM → evaluation.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStoreManager(self.embedding_model)
        self.chunker = SmartTextChunker()
        self.llm = LLMProvider()
        self.evaluator = RAGEvaluator()
        self.memory = ConversationMemory()
        self.multilingual = MultilingualHandler(enabled=True)
        self._ingested_sources: List[str] = []

    # ------------------------------------------------------------------ #
    #  Document Ingestion                                                  #
    # ------------------------------------------------------------------ #

    def ingest_files(self, files: list) -> Dict[str, Any]:
        """Load, chunk, and index uploaded files."""
        docs, errors = load_multiple_documents(files)
        if not docs:
            return {"success": False, "message": "No text extracted.", "errors": errors}
        return self._index_documents(docs, errors)

    def ingest_text(self, text: str, source_name: str = "manual_input") -> Dict[str, Any]:
        """Ingest raw text directly."""
        if not text.strip():
            return {"success": False, "message": "Empty text provided."}
        docs = [{"page_content": text, "metadata": {"source": source_name, "file_type": "text"}}]
        return self._index_documents(docs, [])

    def ingest_web_documents(self, web_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest documents already scraped by WebScraper."""
        if not web_docs:
            return {"success": False, "message": "No web documents provided."}
        return self._index_documents(web_docs, [])

    def _index_documents(
        self, docs: List[Dict[str, Any]], errors: List[str]
    ) -> Dict[str, Any]:
        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            return {"success": False, "message": "Chunking produced no output.", "errors": errors}

        if self.vector_store.is_ready:
            self.vector_store.add_chunks(chunks)
        else:
            self.vector_store.build_from_chunks(chunks)

        for doc in docs:
            src = doc.get("metadata", {}).get("source", "Unknown")
            if src not in self._ingested_sources:
                self._ingested_sources.append(src)

        return {
            "success": True,
            "chunks_added": len(chunks),
            "total_chunks": self.vector_store.document_count,
            "sources": self._ingested_sources,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: int = None,
        search_mode: str = "hybrid",
        evaluate: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG query pipeline.
        Returns answer, retrieved chunks, sources, and evaluation metrics.
        """
        if not self.vector_store.is_ready:
            return {
                "answer": "Please upload and process documents before asking questions.",
                "sources": [],
                "chunks": [],
                "evaluation": None,
                "query_language": "en",
            }

        if not self.llm.is_ready:
            return {
                "answer": "Please configure the LLM (add your API key in Settings).",
                "sources": [],
                "chunks": [],
                "evaluation": None,
                "query_language": "en",
            }

        # 1. Multilingual — detect & translate to English for retrieval
        english_query, detected_lang, lang_confidence = self.multilingual.process_query(question)

        # 2. Contextualise if there is history
        retrieval_query = self._make_standalone_query(english_query)

        # 3. Retrieve top-k chunks
        k = top_k or CONFIG.vector_store.top_k
        retrieved = self.vector_store.retrieve(retrieval_query, top_k=k, mode=search_mode)

        # 4. Build context string
        context_parts = []
        for i, chunk in enumerate(retrieved):
            src = chunk.get("metadata", {}).get("source", "Unknown")
            pg = chunk.get("metadata", {}).get("page", "")
            page_ref = f" (page {pg})" if pg else ""
            context_parts.append(f"[{i+1}] Source: {src}{page_ref}\n{chunk['page_content']}")
        context = "\n\n---\n\n".join(context_parts)

        # 5. Build prompt
        history_str = self.memory.format_history()
        prompt = RAG_SYSTEM_PROMPT.format(
            context=context,
            history=history_str,
            question=english_query,
        )

        # 6. Generate answer
        try:
            answer = self.llm.generate(prompt, chat_history=self.memory.get_history())
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Generation error: {e}"

        # 7. Translate answer back if needed
        if detected_lang != "en":
            answer = self.multilingual.process_answer(answer, detected_lang)

        # 8. Update conversation memory
        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        # 9. Evaluate
        eval_result: Optional[EvaluationResult] = None
        if evaluate and retrieved:
            try:
                contexts_texts = [c["page_content"] for c in retrieved]
                eval_result = self.evaluator.evaluate(english_query, answer, contexts_texts)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        # 10. Compile sources
        sources = list({
            c.get("metadata", {}).get("source", "Unknown") for c in retrieved
        })

        return {
            "answer": answer,
            "sources": sources,
            "chunks": retrieved,
            "evaluation": eval_result,
            "query_language": detected_lang,
            "lang_confidence": lang_confidence,
            "retrieval_mode": search_mode,
            "chunks_retrieved": len(retrieved),
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _make_standalone_query(self, question: str) -> str:
        """Reformulate follow-up questions using chat history."""
        if not self.memory or len(self.memory) == 0:
            return question
        if not CONFIG.conversation.use_chat_history:
            return question
        try:
            history_str = self.memory.format_history()
            prompt = STANDALONE_QUESTION_PROMPT.format(
                history=history_str, question=question
            )
            standalone = self.llm.generate(prompt)
            # Validate that it returned something sensible
            if standalone and len(standalone.strip()) > 10:
                return standalone.strip()
        except Exception:
            pass
        return question

    def reset_conversation(self):
        self.memory.clear()

    def reset_all(self):
        self.vector_store.clear()
        self.memory.clear()
        self._ingested_sources = []

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready and self.llm.is_ready

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "vector_store_ready": self.vector_store.is_ready,
            "llm_ready": self.llm.is_ready,
            "llm_provider": self.llm.provider_name,
            "documents_indexed": self.vector_store.document_count,
            "sources": self.vector_store.get_document_sources(),
            "conversation_turns": len(self.memory) // 2,
        }