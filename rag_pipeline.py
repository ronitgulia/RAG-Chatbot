"""
Core RAG Pipeline — orchestrates all components end-to-end.
"""

import hashlib
import logging
import re
import threading
import time
import uuid
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

# Common English stopwords filtered out during key-term extraction so that
# only meaningful, domain-relevant terms survive.
_STOPWORDS: set = {
    "about", "after", "again", "also", "answer", "based", "because", "before",
    "being", "between", "called", "could", "different", "during", "each",
    "example", "first", "following", "found", "given", "groups", "having",
    "however", "include", "information", "known", "large", "listed", "making",
    "might", "never", "number", "often", "other", "people", "possible",
    "provide", "provided", "question", "rather", "result", "results",
    "second", "several", "should", "simple", "since", "something", "source",
    "still", "support", "system", "their", "there", "these", "thing",
    "things", "those", "though", "through", "under", "using", "various",
    "which", "while", "within", "without", "would",
}


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
        if len(self._history) >= self.max_turns * 2:
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
        # Track content hashes to prevent duplicate chunks.
        self._chunk_hashes: set = set()

        # Async evaluation bookkeeping
        self._eval_results: Dict[str, Optional[EvaluationResult]] = {}
        self._eval_lock = threading.Lock()

        # Restore persisted vector store (survives page refreshes).
        if self.vector_store.load():
            # Rebuild the hash set from previously indexed chunks.
            for doc in self.vector_store._documents:
                h = hashlib.md5(doc["page_content"].encode("utf-8")).hexdigest()
                self._chunk_hashes.add(h)
            # Restore ingested sources list.
            self._ingested_sources = self.vector_store.get_document_sources()
            logger.info(
                f"Restored {self.vector_store.document_count} chunks from cache "
                f"({len(self._ingested_sources)} sources)."
            )

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

        # --- Deduplicate: drop chunks whose content we have already indexed ---
        unique_chunks = []
        for chunk in chunks:
            h = hashlib.md5(chunk["page_content"].encode("utf-8")).hexdigest()
            if h not in self._chunk_hashes:
                self._chunk_hashes.add(h)
                unique_chunks.append(chunk)

        skipped = len(chunks) - len(unique_chunks)
        if skipped:
            logger.info(f"Deduplication: skipped {skipped} duplicate chunk(s).")

        if not unique_chunks:
            return {
                "success": True,
                "chunks_added": 0,
                "total_chunks": self.vector_store.document_count,
                "sources": self._ingested_sources,
                "errors": errors,
                "duplicates_skipped": skipped,
            }

        if self.vector_store.is_ready:
            self.vector_store.add_chunks(unique_chunks)
        else:
            self.vector_store.build_from_chunks(unique_chunks)

        for doc in docs:
            src = doc.get("metadata", {}).get("source", "Unknown")
            if src not in self._ingested_sources:
                self._ingested_sources.append(src)

        # Persist the updated index so it survives page refreshes.
        try:
            self.vector_store.save()
        except Exception as e:
            logger.warning(f"Failed to persist vector store: {e}")

        return {
            "success": True,
            "chunks_added": len(unique_chunks),
            "total_chunks": self.vector_store.document_count,
            "sources": self._ingested_sources,
            "errors": errors,
            "duplicates_skipped": skipped,
        }

    def generate_suggestions(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Generate 5 suggested questions based on the provided documents."""
        if not self.llm.is_ready or not docs:
            return []
            
        # Sample text to give context (max ~2000 chars to save tokens/time)
        context_parts = []
        char_count = 0
        for d in docs:
            text = d.get("page_content", "")
            if char_count + len(text) > 2000:
                text = text[:2000 - char_count]
            if text:
                context_parts.append(text)
                char_count += len(text)
            if char_count >= 2000:
                break
                
        context = "\n".join(context_parts)
        if not context.strip():
            return []

        prompt = (
            "You are an AI assistant. Based on the following document excerpt, generate exactly 5 interesting "
            "and diverse questions that a user might want to ask about this document. "
            "Output ONLY the questions, one per line, with no numbering, bullet points, or introductory text.\n\n"
            f"Document Excerpt:\n{context}\n\nQuestions:"
        )

        try:
            response = self.llm.generate(prompt)
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                # Remove common numbering/bullets if the LLM adds them anyway
                line = re.sub(r'^[\d\-\*\.\)]+\s*', '', line)
                if line and len(line) > 5 and "?" in line:
                    questions.append(line)
            
            return questions[:5]
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []

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

        # 9. Evaluate — run in a background thread and join with a short
        #    timeout so fast heuristic scores are returned inline while slow
        #    RAGAS calls don't block the user.
        eval_result: Optional[EvaluationResult] = None
        eval_id: Optional[str] = None
        if evaluate and retrieved:
            eval_id = uuid.uuid4().hex[:8]
            contexts_texts = [c["page_content"] for c in retrieved]
            # Use a sentinel object so we can tell "still running" from
            # "finished with None" once the lock is released.
            _PENDING = object()
            with self._eval_lock:
                self._eval_results[eval_id] = _PENDING
            t = threading.Thread(
                target=self._run_eval_background,
                args=(eval_id, english_query, answer, contexts_texts),
                daemon=True,
            )
            t.start()
            # Give the thread up to 2 seconds — enough for the heuristic
            # evaluator (< 10 ms) but not long enough to block on RAGAS.
            t.join(timeout=2.0)
            # Harvest the result if the thread finished within the window.
            with self._eval_lock:
                finished = self._eval_results.get(eval_id)
                if finished is not _PENDING and finished is not None:
                    eval_result = finished
                    del self._eval_results[eval_id]  # free memory immediately
                    eval_id = None  # no need for caller to poll

        # 10. Compile sources
        sources = list({
            c.get("metadata", {}).get("source", "Unknown") for c in retrieved
        })

        return {
            "answer": answer,
            "sources": sources,
            "chunks": retrieved,
            "evaluation": eval_result,   # populated inline for fast evals
            "eval_id": eval_id,          # non-None only when still pending
            "query_language": detected_lang,
            "lang_confidence": lang_confidence,
            "retrieval_mode": search_mode,
            "chunks_retrieved": len(retrieved),
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Async Evaluation helpers                                            #
    # ------------------------------------------------------------------ #

    def _run_eval_background(
        self, eval_id: str, question: str, answer: str, contexts: List[str]
    ):
        """Run evaluation in a daemon thread and stash the result.

        The slot in ``_eval_results`` is pre-populated with a sentinel object
        by the caller; we overwrite it with the real result (or ``None`` on
        failure) so that ``get_eval_result`` can distinguish *still running*
        from *completed with no result*.
        """
        try:
            result = self.evaluator.evaluate(question, answer, contexts)
        except Exception as e:
            logger.warning(f"Background eval failed: {e}")
            result = None
        with self._eval_lock:
            # Only write if the slot still belongs to us (caller may have
            # already harvested and deleted it after the join timeout).
            if eval_id in self._eval_results:
                self._eval_results[eval_id] = result

    def get_eval_result(self, eval_id: str) -> Optional[EvaluationResult]:
        """Return the evaluation result for *eval_id*, or ``None`` if still
        pending.  A completed result is removed from the internal dict to
        prevent unbounded memory growth.
        """
        if eval_id is None:
            return None
        with self._eval_lock:
            slot = self._eval_results.get(eval_id)
            # Key absent → unknown id.
            if eval_id not in self._eval_results:
                return None
            # If the slot contains a sentinel object (not None, not EvaluationResult), it is still pending.
            if slot is not None and not isinstance(slot, EvaluationResult):
                return None
            # Concrete result available or evaluation failed (None) — harvest and free.
            del self._eval_results[eval_id]
            return slot

    # ------------------------------------------------------------------ #
    #  Lightweight query rewrite (no LLM call)                             #
    # ------------------------------------------------------------------ #

    # Words that suggest the question references prior conversation context.
    _CONTEXT_CUE_WORDS = {
        "it", "its", "they", "them", "their", "theirs",
        "this", "that", "these", "those",
        "he", "she", "his", "her", "hers",
        "above", "previous", "earlier", "before",
        "same", "also", "too", "again", "more",
    }

    @staticmethod
    def _looks_like_followup(question: str) -> bool:
        """Return True if the question appears to reference prior context."""
        words = set(question.lower().split())
        return bool(words & RAGPipeline._CONTEXT_CUE_WORDS)

    def _get_last_assistant_message(self) -> Optional[str]:
        """Retrieve the most recent assistant message from memory."""
        for msg in reversed(self.memory.get_history()):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    @staticmethod
    def _extract_key_terms(text: str, max_terms: int = 6) -> List[str]:
        """Extract salient terms from *text* using cheap regex heuristics.

        Priority order: capitalised phrases → quoted terms → long words.
        Common stopwords are filtered out.
        """
        # Capitalised multi-word phrases (proper nouns, titles)
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
        # Longer words (likely domain-specific)
        words = re.findall(r"\b[a-zA-Z]{6,}\b", text)

        seen: set = set()
        result: List[str] = []
        for term in caps + quoted + words:
            key = term.lower().strip()
            if key not in seen and key not in _STOPWORDS:
                seen.add(key)
                result.append(term)
            if len(result) >= max_terms:
                break
        return result

    def _make_standalone_query(self, question: str) -> str:
        """Contextualise follow-up questions using chat history.

        Uses a zero-cost rule-based heuristic instead of an LLM call:
        extracts key terms from the last assistant response and prepends
        them to the query, giving the retriever the extra keywords it
        needs without burning an API call or risking rate limits.
        """
        if not self.memory or len(self.memory) == 0:
            return question
        if not CONFIG.conversation.use_chat_history:
            return question
        if not self._looks_like_followup(question):
            return question

        last_response = self._get_last_assistant_message()
        if not last_response:
            return question

        key_terms = self._extract_key_terms(last_response, max_terms=6)
        if not key_terms:
            return question

        rewritten = f"Regarding {', '.join(key_terms)}: {question}"
        logger.info(f"Query rewrite (rule-based): {rewritten}")
        return rewritten

    def update_chunker_settings(
        self, chunk_size: int, chunk_overlap: int, strategy: str = "recursive"
    ):
        """Forward chunking settings to the underlying SmartTextChunker."""
        self.chunker.update_settings(chunk_size, chunk_overlap, strategy)

    def reset_conversation(self):
        self.memory.clear()

    def reset_all(self):
        self.vector_store.clear()
        self.memory.clear()
        self._ingested_sources = []
        self._chunk_hashes.clear()
        # Delete the persisted cache files.
        import os
        import shutil
        cache_path = "vector_store_cache.pkl"
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info(f"Deleted vector store cache: {cache_path}")
            except OSError as e:
                logger.warning(f"Could not delete cache file: {e}")
                
        faiss_dir = "faiss_index"
        if os.path.exists(faiss_dir):
            try:
                shutil.rmtree(faiss_dir)
                logger.info(f"Deleted faiss index directory: {faiss_dir}")
            except OSError as e:
                logger.warning(f"Could not delete faiss index directory: {e}")

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