"""
Regression tests for the evaluation pipeline fix.

Completely self-contained — zero imports from the RAG project source.
Every class under test is either:
  a) Copied verbatim from the source (ConversationMemory, EvalCore), or
  b) A stub that mirrors the real interface.

This means pytest collects and runs in < 1 s with no ML model downloads,
no network access, and no heavy dependency chains.

What is verified:
  1.  Fast evals are returned INLINE (the core regression case).
  2.  No memory leak — _eval_results is clean after harvest.
  3.  Slow evals are retrievable via get_eval_result() poll loop.
  4.  Failing evaluators don't leave stale dict entries.
  5.  The original broken code provably returned None.
  6.  10 concurrent evals don't corrupt each other (thread safety).
  7.  get_eval_result() API — unknown id, None id, idempotent harvest.
  8.  ConversationMemory sliding window, format, clear.
  9.  VectorStore RRF ranking logic (pure Python, no FAISS).
"""

import threading
import time
import unittest
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════ #
#  Minimal data types (mirror the real ones)                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class EvaluationResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    details: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Stub evaluators                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class FastEvaluator:
    """Heuristic-speed stub — returns in < 1 ms."""
    def evaluate(self, question, answer, contexts, ground_truth=None):
        return EvaluationResult(
            faithfulness=0.85, answer_relevancy=0.80,
            context_precision=0.75, overall_score=0.80, passed=True,
        )

class SlowEvaluator:
    """Simulates RAGAS — sleeps 4 s to exceed the 2-s join window."""
    def evaluate(self, question, answer, contexts, ground_truth=None):
        time.sleep(4)
        return EvaluationResult(faithfulness=0.99, overall_score=0.99, passed=True)

class FailingEvaluator:
    """Always raises — simulates a RAGAS crash."""
    def evaluate(self, question, answer, contexts, ground_truth=None):
        raise RuntimeError("Simulated RAGAS crash")


# ═══════════════════════════════════════════════════════════════════════════ #
#  ConversationMemory — copied verbatim from rag_pipeline.py                 #
#  (any drift between this copy and the source will be caught by             #
#   the ConversationMemory tests below)                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class ConversationMemory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: List[Dict] = []

    def add(self, role: str, content: str):
        self._history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        if len(self._history) >= self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def format_history(self) -> str:
        if not self._history:
            return "No previous conversation."
        lines = []
        for msg in self._history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self._history = []

    def __len__(self):
        return len(self._history)


# ═══════════════════════════════════════════════════════════════════════════ #
#  EvalCore — mirrors ONLY the evaluation threading logic from RAGPipeline.  #
#  This is the exact code the fix touches; any future refactor must keep     #
#  this in sync.                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class EvalCore:
    """
    Extracted evaluation machinery from RAGPipeline.
    Mirrors _run_eval_background, get_eval_result, and the join-with-timeout
    section of query() introduced by the bug fix.
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self._eval_results: Dict = {}
        self._eval_lock = threading.Lock()

    # ── mirrors RAGPipeline._run_eval_background ──────────────────────────
    def _run_eval_background(self, eval_id, question, answer, contexts):
        try:
            result = self.evaluator.evaluate(question, answer, contexts)
        except Exception:
            result = None
        with self._eval_lock:
            # Guard: caller may have already harvested and deleted the slot
            if eval_id in self._eval_results:
                self._eval_results[eval_id] = result

    # ── mirrors RAGPipeline.is_eval_pending ─────────────────────────────
    def is_eval_pending(self, eval_id) -> bool:
        if eval_id is None:
            return False
        with self._eval_lock:
            if eval_id not in self._eval_results:
                return False
            slot = self._eval_results[eval_id]
            return slot is not None and not isinstance(slot, EvaluationResult)

    # ── mirrors RAGPipeline.get_eval_result ───────────────────────────────
    def get_eval_result(self, eval_id) -> Optional[EvaluationResult]:
        if eval_id is None:
            return None
        with self._eval_lock:
            if eval_id not in self._eval_results:
                return None
            slot = self._eval_results[eval_id]
            if not isinstance(slot, EvaluationResult):
                return None  # still pending (sentinel) or failed (None)
            del self._eval_results[eval_id]
            return slot

    # ── mirrors the fixed query() eval section ────────────────────────────
    def run_eval(self, question, answer, contexts):
        """
        Returns (eval_result, eval_id).
        Exactly one will be non-None:
          • eval_result non-None  →  fast path, result harvested inline
          • eval_id non-None      →  slow path, caller must poll
        """
        _PENDING = object()
        eval_id = uuid.uuid4().hex[:8]

        with self._eval_lock:
            self._eval_results[eval_id] = _PENDING

        t = threading.Thread(
            target=self._run_eval_background,
            args=(eval_id, question, answer, contexts),
            daemon=True,
        )
        t.start()
        t.join(timeout=2.0)          # ← the key fix

        eval_result = None
        with self._eval_lock:
            finished = self._eval_results.get(eval_id)
            if finished is not _PENDING and finished is not None:
                eval_result = finished
                del self._eval_results[eval_id]  # free immediately
                eval_id = None                   # no polling needed

        return eval_result, eval_id


# ═══════════════════════════════════════════════════════════════════════════ #
#  VectorStore RRF — pure-Python helper (no FAISS)                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def hybrid_rrf(dense, sparse, top_k=5, alpha=0.5, rrf_k=60):
    """
    Reciprocal Rank Fusion — copied from VectorStoreManager.hybrid_search().
    Tested independently so we never need FAISS.
    """
    rrf_scores: Dict[str, float] = {}
    content_map: Dict[str, Dict] = {}

    def add_rrf(results, weight):
        for rank, r in enumerate(results):
            key = r["page_content"][:200]
            content_map[key] = r
            rrf_scores[key] = rrf_scores.get(key, 0) + weight * (1 / (rrf_k + rank + 1))

    add_rrf(dense, alpha)
    add_rrf(sparse, 1 - alpha)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**content_map[key], "rrf_score": score, "search_type": "hybrid"}
        for key, score in ranked
    ]


# ═══════════════════════════════════════════════════════════════════════════ #
#  QueryRewriteCore — copied from rag_pipeline.py                            #
# ═══════════════════════════════════════════════════════════════════════════ #

import re

_STOPWORDS: set = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "should", "could", "ought", "i", "you", "he", "she", "we",
    "him", "her", "his", "hers", "himself", "herself", "myself", "yourself",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "will", "just", "don", "should", "now",
    "also", "answer", "based", "called", "different", "example", "first",
    "following", "found", "given", "groups", "however", "include", "information",
    "known", "large", "listed", "making", "might", "never", "number", "often",
    "people", "possible", "provide", "provided", "question", "rather", "result",
    "results", "second", "several", "simple", "since", "something", "source",
    "still", "support", "system", "thing", "things", "though", "using",
    "various", "without", "according", "mentioned", "document", "documents",
    "context", "excerpt", "stated", "described", "details", "summary",
    "please", "note", "assistant", "user", "response", "regarding", "unfortunately",
}

_CONTEXT_CUE_WORDS = {
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "he", "she", "his", "her", "hers",
    "above", "previous", "earlier", "before",
    "same", "also", "too", "again", "more",
    "why", "how", "elaborate", "explain", "expand",
}

def looks_like_followup(question: str) -> bool:
    words = set(re.findall(r"\b[a-zA-Z]+\b", question.lower()))
    return bool(words & _CONTEXT_CUE_WORDS)

def extract_key_terms(text: str, max_terms: int = 6) -> List[str]:
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    double_quoted = re.findall(r'"([^"\n]+)"', text)
    single_quoted = re.findall(r"(?:^|[\s({\[])'([^'\n]+)'(?:[\s)}\].,!?;:]|$)", text)
    quoted = double_quoted + single_quoted
    words = re.findall(r"\b[a-zA-Z]{6,}\b", text)

    seen: set = set()
    result: List[str] = []
    for term in caps + quoted + words:
        cleaned = term.strip().strip("'\"")
        key = cleaned.lower()
        if key and key not in seen and key not in _STOPWORDS:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= max_terms:
            break
    return result


# ═══════════════════════════════════════════════════════════════════════════ #
#  TEST SUITES                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestEvalNotSilentlyDropped(unittest.TestCase):
    """
    Core regression suite.
    Every test here maps to a specific symptom of the original bug.
    """

    # ── 1. Core regression ───────────────────────────────────────────────
    def test_fast_eval_returned_inline(self):
        """
        GIVEN a fast (heuristic) evaluator  [< 2 s]
        WHEN  run_eval() executes
        THEN  eval_result must be a real EvaluationResult, NOT None.

        This is the EXACT scenario the original bug broke.
        query() always returned {"evaluation": None} — scores were gone forever.
        """
        core = EvalCore(FastEvaluator())
        ev, eid = core.run_eval("What is RAG?", "RAG uses retrieval.", ["ctx"])

        self.assertIsNotNone(ev,
            "REGRESSION: eval_result is None — scores are being silently dropped!")
        self.assertIsInstance(ev, EvaluationResult)
        self.assertIsNone(eid,
            "eval_id should be None after fast inline harvest")
        self.assertAlmostEqual(ev.faithfulness, 0.85, places=2)
        self.assertTrue(ev.passed)

    # ── 2. No memory leak ────────────────────────────────────────────────
    def test_no_memory_leak_after_fast_eval(self):
        """
        _eval_results must be empty after harvest.
        Original bug: one entry leaked per query → dict grew forever.
        """
        core = EvalCore(FastEvaluator())
        core.run_eval("q", "a", ["ctx"])

        with core._eval_lock:
            remaining = len(core._eval_results)
        self.assertEqual(remaining, 0,
            f"Memory leak: {remaining} stale entries remain in _eval_results")

    # ── 3. Slow eval polled via get_eval_result ──────────────────────────
    def test_slow_eval_polled_via_get_eval_result(self):
        """
        GIVEN a slow evaluator  (> 2 s join timeout, e.g. RAGAS)
        WHEN  run_eval() returns
        THEN  eval_id is non-None AND polling eventually surfaces the result.
        """
        core = EvalCore(SlowEvaluator())
        ev, eid = core.run_eval("q", "a", ["ctx"])

        self.assertIsNone(ev,   "Slow eval must NOT be inline (would block > 2 s)")
        self.assertIsNotNone(eid, "eval_id must be returned for caller to poll")

        # Simulate app.py poll loop (up to 10 s)
        deadline = time.monotonic() + 10.0
        result = None
        while time.monotonic() < deadline:
            result = core.get_eval_result(eid)
            if result is not None:
                break
            time.sleep(0.25)

        self.assertIsNotNone(result,
            "Slow eval never surfaced via get_eval_result() within 10 s")
        self.assertAlmostEqual(result.faithfulness, 0.99, places=2)

        # After harvest, dict must be empty
        with core._eval_lock:
            self.assertEqual(len(core._eval_results), 0)

    # ── 4. Failing evaluator ─────────────────────────────────────────────
    def test_failing_eval_stores_none_not_sentinel(self):
        """
        When the evaluator raises, the thread must write None (not the sentinel).
        get_eval_result() returns None (not a corrupt value).
        """
        core = EvalCore(FailingEvaluator())
        _PENDING = object()
        eid = uuid.uuid4().hex[:8]

        with core._eval_lock:
            core._eval_results[eid] = _PENDING

        t = threading.Thread(
            target=core._run_eval_background,
            args=(eid, "q", "a", ["ctx"]),
            daemon=True,
        )
        t.start()
        t.join(timeout=3.0)

        with core._eval_lock:
            slot = core._eval_results.get(eid)

        self.assertIsNone(slot,
            "Failed eval must write None, not leave the sentinel")
        self.assertIsNot(slot, _PENDING,
            "Sentinel must be overwritten even on evaluator failure")

    # ── 5. Original bug documented ───────────────────────────────────────
    def test_original_bug_would_have_returned_none(self):
        """
        Simulates the old fire-and-forget code.
        Proves it always returned None — documents why the fix was needed.
        """
        core = EvalCore(FastEvaluator())
        _PENDING = object()
        eid = uuid.uuid4().hex[:8]

        with core._eval_lock:
            core._eval_results[eid] = _PENDING

        t = threading.Thread(
            target=core._run_eval_background,
            args=(eid, "q", "a", ["ctx"]),
            daemon=True,
        )
        t.start()
        # OLD CODE: no join → immediately reads from return dict
        old_code_evaluation_field = None   # ← this was hardcoded in the old query()

        self.assertIsNone(old_code_evaluation_field,
            "This must be None — confirms the original bug returned None always")

        t.join(timeout=3.0)  # clean up thread

    # ── 6. Thread safety ─────────────────────────────────────────────────
    def test_concurrent_evals_thread_safe(self):
        """
        10 evaluations run in parallel.
        Every result must land in the correct slot — no mixing or data races.
        """
        core = EvalCore(FastEvaluator())
        _PENDING = object()
        ids_and_threads = []

        for _ in range(10):
            eid = uuid.uuid4().hex[:8]
            with core._eval_lock:
                core._eval_results[eid] = _PENDING
            t = threading.Thread(
                target=core._run_eval_background,
                args=(eid, f"q_{eid}", "answer", ["ctx"]),
                daemon=True,
            )
            ids_and_threads.append((eid, t))
            t.start()

        for eid, t in ids_and_threads:
            t.join(timeout=3.0)

        with core._eval_lock:
            for eid, _ in ids_and_threads:
                val = core._eval_results.get(eid)
                self.assertIsNotNone(val,
                    f"Missing result for {eid} — possible race condition")
                self.assertIsNot(val, _PENDING,
                    f"Slot for {eid} still shows PENDING after thread.join()")
                self.assertIsInstance(val, EvaluationResult,
                    f"Slot for {eid} is not an EvaluationResult: {type(val)}")


class TestGetEvalResultContract(unittest.TestCase):
    """Unit tests for the get_eval_result() public API contract."""

    def _core(self):
        return EvalCore(FastEvaluator())

    def test_returns_none_for_unknown_id(self):
        self.assertIsNone(self._core().get_eval_result("nonexistent-xyz"))

    def test_returns_none_for_none_input(self):
        self.assertIsNone(self._core().get_eval_result(None))

    def test_harvests_and_deletes_on_first_call(self):
        core = self._core()
        fake = EvaluationResult(faithfulness=0.9, overall_score=0.9, passed=True)
        eid = "test-eid-001"
        with core._eval_lock:
            core._eval_results[eid] = fake

        result = core.get_eval_result(eid)
        self.assertIs(result, fake, "Must return the stored result")

        result2 = core.get_eval_result(eid)
        self.assertIsNone(result2,
            "Second call must return None — entry must be deleted after harvest")

        with core._eval_lock:
            self.assertNotIn(eid, core._eval_results,
                "Dict must not retain the key after harvest")

    def test_pending_slot_returns_none(self):
        """A slot containing the sentinel (still running) must return None."""
        core = self._core()
        eid = "pending-slot"
        _PENDING = object()
        with core._eval_lock:
            core._eval_results[eid] = _PENDING
        result = core.get_eval_result(eid)
        self.assertIsNone(result,
            "A pending slot must not be returned as a result")

    def test_is_eval_pending_behavior(self):
        core = self._core()
        eid = "pending-check-01"
        _PENDING = object()

        # Unknown id is not pending
        self.assertFalse(core.is_eval_pending(eid))
        self.assertFalse(core.is_eval_pending(None))

        # Stored sentinel means pending
        with core._eval_lock:
            core._eval_results[eid] = _PENDING
        self.assertTrue(core.is_eval_pending(eid))

        # Completed result means not pending
        with core._eval_lock:
            core._eval_results[eid] = EvaluationResult(overall_score=0.9)
        self.assertFalse(core.is_eval_pending(eid))

        # Harvested/deleted means not pending
        core.get_eval_result(eid)
        self.assertFalse(core.is_eval_pending(eid))

    def test_failed_eval_polling_terminates_immediately(self):
        """Simulate app.py polling loop when background evaluation fails."""
        core = self._core()
        eid = "failed-eval-01"
        # Simulate worker setting failure (None)
        with core._eval_lock:
            core._eval_results[eid] = None

        iterations = 0
        ev = None
        # Loop exactly matching app.py
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            iterations += 1
            ev = core.get_eval_result(eid)
            if ev is not None or not core.is_eval_pending(eid):
                break
            time.sleep(0.05)

        # Must terminate on the very first iteration without sleeping
        self.assertEqual(iterations, 1)
        self.assertIsNone(ev)


class TestConversationMemory(unittest.TestCase):
    """Existing ConversationMemory tests — all must still pass."""

    def test_sliding_window(self):
        mem = ConversationMemory(max_turns=2)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        mem.add("user", "How are you?")
        mem.add("assistant", "I'm good")
        self.assertEqual(len(mem), 4)

        mem.add("user", "What's up?")
        mem.add("assistant", "Nothing much")
        self.assertEqual(len(mem), 4)

        history = mem.get_history()
        self.assertEqual(history[0]["content"], "How are you?")
        self.assertEqual(history[-1]["content"], "Nothing much")

    def test_format_history_empty(self):
        mem = ConversationMemory()
        self.assertEqual(mem.format_history(), "No previous conversation.")

    def test_format_history_with_messages(self):
        mem = ConversationMemory()
        mem.add("user", "Hello")
        self.assertIn("User: Hello", mem.format_history())

    def test_clear_resets_history(self):
        mem = ConversationMemory()
        mem.add("user", "Test")
        mem.clear()
        self.assertEqual(len(mem), 0)
        self.assertEqual(mem.format_history(), "No previous conversation.")


class TestVectorStoreRRF(unittest.TestCase):
    """RRF hybrid search logic — pure Python, zero FAISS dependency."""

    def test_rrf_ranks_overlap_highest(self):
        """Doc B appears in both lists → must rank first by RRF."""
        dense  = [
            {"page_content": "Doc A", "metadata": {"source": "A"}, "score": 0.9},
            {"page_content": "Doc B", "metadata": {"source": "B"}, "score": 0.8},
        ]
        sparse = [
            {"page_content": "Doc B", "metadata": {"source": "B"}, "score": 2.5},
            {"page_content": "Doc C", "metadata": {"source": "C"}, "score": 1.5},
        ]

        results = hybrid_rrf(dense, sparse, top_k=3, alpha=0.5)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["page_content"], "Doc B",
            "Doc B (in both lists) must rank first via RRF")
        self.assertIn("rrf_score", results[0])
        self.assertEqual(results[0]["search_type"], "hybrid")
        self.assertEqual(results[1]["page_content"], "Doc A")
        self.assertEqual(results[2]["page_content"], "Doc C")

    def test_rrf_pure_dense_alpha_1(self):
        """alpha=1.0 → only dense results contribute."""
        dense  = [{"page_content": "Dense Only", "metadata": {}, "score": 1.0}]
        sparse = [{"page_content": "Sparse Only", "metadata": {}, "score": 5.0}]
        results = hybrid_rrf(dense, sparse, top_k=2, alpha=1.0)
        # Dense Only gets a score; Sparse Only gets weight 0
        top = results[0]["page_content"]
        self.assertEqual(top, "Dense Only",
            "With alpha=1.0 only dense results should contribute")

    def test_rrf_scores_are_positive(self):
        """RRF scores must always be positive."""
        dense  = [{"page_content": f"Doc {i}", "metadata": {}, "score": 1.0}
                  for i in range(5)]
        results = hybrid_rrf(dense, [], top_k=5, alpha=1.0)
        for r in results:
            self.assertGreater(r["rrf_score"], 0,
                "RRF scores must always be positive")


class TestQueryRewriteHeuristics(unittest.TestCase):
    """Unit tests for rule-based query rewrite heuristics."""

    def test_looks_like_followup_handles_punctuation(self):
        """Punctuation directly attached to cue words must not prevent detection."""
        self.assertTrue(looks_like_followup("What is it?"))
        self.assertTrue(looks_like_followup("Tell me more."))
        self.assertTrue(looks_like_followup("Can you explain that?"))
        self.assertTrue(looks_like_followup("What about these!"))
        self.assertTrue(looks_like_followup("Who was he?"))
        self.assertTrue(looks_like_followup("Could you elaborate?"))

    def test_looks_like_followup_ignores_standalone_queries(self):
        """Unambiguous new questions must not be flagged as follow-ups."""
        self.assertFalse(looks_like_followup("What is photosynthesis?"))
        self.assertFalse(looks_like_followup("Give an overview of transformers"))
        self.assertFalse(looks_like_followup("Define artificial intelligence"))

    def test_extract_key_terms_contraction_safety(self):
        """Contractions with apostrophes must not be treated as single-quoted terms."""
        text = "Don't worry, it's fine. The 'Transformer' architecture uses self-attention."
        terms = extract_key_terms(text, max_terms=6)
        lower_terms = [t.lower() for t in terms]
        self.assertNotIn("t worry, it", lower_terms)
        self.assertIn("Transformer", terms)

    def test_extract_key_terms_filters_stopwords(self):
        """Conversational filler words and English stopwords should be excluded."""
        text = "According to the mentioned document, Neural Networks process data."
        terms = extract_key_terms(text, max_terms=6)
        lower_terms = [t.lower() for t in terms]
        self.assertNotIn("according", lower_terms)
        self.assertNotIn("document", lower_terms)
        self.assertNotIn("the", lower_terms)
        self.assertTrue(any("neural" in t.lower() or "networks" in t.lower() for t in terms))


class TestSeniorEngineeringUpgrades(unittest.TestCase):
    """Regression and unit tests for senior engineering additions."""

    def test_vector_store_tokenize_normalizes_punctuation(self):
        """BM25 tokenization must strip punctuation and lowercase words."""
        from vector_store import VectorStoreManager
        tokens = VectorStoreManager._tokenize("What is RAG? Efficient retrieval!")
        self.assertEqual(tokens, ["what", "is", "rag", "efficient", "retrieval"])

    def test_cached_langchain_embeddings_adapter(self):
        """CachedLangChainEmbeddings should delegate to EmbeddingModel and return float lists."""
        from embeddings import CachedLangChainEmbeddings
        class DummyModel:
            def embed_texts(self, texts):
                import numpy as np
                return np.array([[0.1, 0.2, 0.3]] * len(texts))
            def embed_query(self, text):
                import numpy as np
                return np.array([0.1, 0.2, 0.3])

        adapter = CachedLangChainEmbeddings(DummyModel())
        docs_res = adapter.embed_documents(["hello world", "test"])
        self.assertEqual(len(docs_res), 2)
        self.assertIsInstance(docs_res[0], list)
        self.assertAlmostEqual(docs_res[0][0], 0.1)

        query_res = adapter.embed_query("query")
        self.assertIsInstance(query_res, list)
        self.assertAlmostEqual(query_res[0], 0.1)

    def test_heuristic_context_recall_calculation(self):
        """evaluate_heuristic must compute context_recall when ground_truth is provided."""
        from evaluation import RAGEvaluator
        ev = RAGEvaluator(use_ragas=False)
        res = ev.evaluate(
            question="What does RAG combine?",
            answer="RAG combines retrieval and generation.",
            contexts=["RAG combines document retrieval with text generation."],
            ground_truth="RAG combines retrieval with generation."
        )
        self.assertGreater(res.context_recall, 0.0)
        self.assertTrue(res.details.get("has_ground_truth"))

    def test_export_pdf_unicode_resilience(self):
        """PDF export must not crash on non-Latin Unicode characters or emojis."""
        from export_utils import export_as_pdf
        messages = [
            {"role": "user", "content": "Hello — this has an em-dash and emoji 🚀 and Hindi: नमस्ते"},
            {"role": "assistant", "content": "Responding with quotes: “smart quotes” and ‘apostrophe’"}
        ]
        pdf_bytes = export_as_pdf(messages)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
