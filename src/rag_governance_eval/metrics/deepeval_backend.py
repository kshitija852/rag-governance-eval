from __future__ import annotations

from typing import List, Optional

from .base import MetricBackend, MetricScores, BackendUnavailableError

try:
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        HallucinationMetric,
        AnswerRelevancyMetric,
        ToxicityMetric,
        BiasMetric,
        PIILeakageMetric,
    )

    _IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    _IMPORT_ERROR = exc


class DeepEvalBackend(MetricBackend):
    name = "deepeval"

    def __init__(
        self,
        model: Optional[str] = None,
        threshold: float = 0.5,
        enable_safety_metrics: bool = False,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise BackendUnavailableError(
                f"deepeval is not importable. Original error: {_IMPORT_ERROR!r}"
            )
        self.model = model
        self.threshold = threshold
        self.enable_safety_metrics = enable_safety_metrics

    def score(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> MetricScores:
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=contexts,
            expected_output=reference,
            context=contexts,
        )

        faithfulness = FaithfulnessMetric(threshold=self.threshold, model=self.model)
        faithfulness.measure(test_case)

        hallucination = HallucinationMetric(threshold=self.threshold, model=self.model)
        hallucination.measure(test_case)

        answer_relevancy = AnswerRelevancyMetric(threshold=self.threshold, model=self.model)
        answer_relevancy.measure(test_case)

        context_precision = ContextualPrecisionMetric(threshold=self.threshold, model=self.model)
        context_recall = ContextualRecallMetric(threshold=self.threshold, model=self.model)

        if reference is not None:
            context_precision.measure(test_case)
            context_recall.measure(test_case)
            precision_score = context_precision.score
            recall_score = context_recall.score
        else:
            precision_score = None
            recall_score = None

        toxicity_score = bias_score = pii_score = None
        if self.enable_safety_metrics:
            toxicity = ToxicityMetric(threshold=self.threshold, model=self.model)
            toxicity.measure(test_case)
            toxicity_score = toxicity.score

            bias = BiasMetric(threshold=self.threshold, model=self.model)
            bias.measure(test_case)
            bias_score = bias.score

            pii = PIILeakageMetric(threshold=self.threshold, model=self.model)
            pii.measure(test_case)
            pii_score = pii.score

        return MetricScores(
            faithfulness=faithfulness.score,
            hallucination=hallucination.score,
            context_precision=precision_score,
            context_recall=recall_score,
            answer_relevancy=answer_relevancy.score,
            toxicity=toxicity_score,
            bias=bias_score,
            pii_leakage=pii_score,
        )