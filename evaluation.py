"""
RAG Evaluation using RAGAS framework.
Metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics.
    Falls back to lightweight heuristic scoring if RAGAS is not available.
    """

    def __init__(self, use_ragas: bool = True):
        self.use_ragas = use_ragas
        self._ragas_available = self._check_ragas()

    @staticmethod
    def _check_ragas() -> bool:
        try:
            import ragas
            return True
        except ImportError:
            logger.warning("RAGAS not available; using heuristic evaluation.")
            return False

    # ------------------------------------------------------------------ #
    #  RAGAS evaluation                                                    #
    # ------------------------------------------------------------------ #

    def evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run RAGAS evaluation pipeline.
        ground_truth is optional — skipped metrics are set to 0.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            metrics = [faithfulness, answer_relevancy, context_precision]

            result = evaluate(dataset, metrics=metrics)
            result_dict = result.to_pandas().to_dict(orient="records")[0]

            f = float(result_dict.get("faithfulness", 0))
            ar = float(result_dict.get("answer_relevancy", 0))
            cp = float(result_dict.get("context_precision", 0))
            overall = round((f + ar + cp) / 3, 3)

            return EvaluationResult(
                faithfulness=round(f, 3),
                answer_relevancy=round(ar, 3),
                context_precision=round(cp, 3),
                overall_score=overall,
                passed=overall >= 0.65,
                details=result_dict,
            )
        except Exception as e:
            logger.error(f"RAGAS evaluation error: {e}")
            return self.evaluate_heuristic(question, answer, contexts)

    # ------------------------------------------------------------------ #
    #  Heuristic fallback                                                  #
    # ------------------------------------------------------------------ #

    def evaluate_heuristic(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> EvaluationResult:
        """
        Lightweight heuristic scoring (no external API calls).
        Suitable when RAGAS or an LLM judge is not available.
        """
        import re

        def word_overlap(a: str, b: str) -> float:
            wa = set(re.findall(r"\w+", a.lower()))
            wb = set(re.findall(r"\w+", b.lower()))
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / len(wa | wb)

        # Faithfulness: answer words that appear in any context
        combined_ctx = " ".join(contexts)
        faithfulness = min(1.0, word_overlap(answer, combined_ctx) * 2)

        # Answer relevancy: answer words overlapping with question
        answer_relevancy = min(1.0, word_overlap(answer, question) * 3)

        # Context precision: question words appearing in retrieved contexts
        ctx_scores = [word_overlap(question, c) for c in contexts]
        context_precision = round(sum(ctx_scores) / max(len(ctx_scores), 1), 3)

        overall = round((faithfulness + answer_relevancy + context_precision) / 3, 3)

        return EvaluationResult(
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(answer_relevancy, 3),
            context_precision=round(context_precision, 3),
            overall_score=overall,
            passed=overall >= 0.5,
            details={"method": "heuristic"},
        )

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        if self.use_ragas and self._ragas_available:
            return self.evaluate_with_ragas(question, answer, contexts, ground_truth)
        return self.evaluate_heuristic(question, answer, contexts)


# ------------------------------------------------------------------ #
#  Batch evaluation helper                                             #
# ------------------------------------------------------------------ #

def batch_evaluate(
    evaluator: RAGEvaluator,
    qa_pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of QA pairs.
    Each item should have: question, answer, contexts, optional ground_truth.
    """
    results = []
    for pair in qa_pairs:
        result = evaluator.evaluate(
            question=pair["question"],
            answer=pair["answer"],
            contexts=pair["contexts"],
            ground_truth=pair.get("ground_truth"),
        )
        results.append({
            "question": pair["question"],
            "answer": pair["answer"][:120] + "...",
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "overall_score": result.overall_score,
            "passed": result.passed,
        })
    return results