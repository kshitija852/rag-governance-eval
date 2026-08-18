from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MetricScores:
    faithfulness: float
    hallucination: float
    context_precision: Optional[float]
    context_recall: Optional[float]
    answer_relevancy: Optional[float] = None  # is the answer addressing
    # what was actually asked - independent of faithfulness (a faithful
    # answer can still dodge the question)
    toxicity: Optional[float] = None  # opt-in (enable_safety_metrics) -
    # None means "not computed", not "clean"
    bias: Optional[float] = None  # opt-in, same convention
    pii_leakage: Optional[float] = None  # opt-in, same convention

    def as_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "hallucination": self.hallucination,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_relevancy": self.answer_relevancy,
            "toxicity": self.toxicity,
            "bias": self.bias,
            "pii_leakage": self.pii_leakage,
        }


class MetricBackend(ABC):
    name: str = "base"

    @abstractmethod
    def score(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: str | None = None,
    ) -> MetricScores:
        raise NotImplementedError


class BackendUnavailableError(RuntimeError):
    pass