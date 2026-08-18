"""
Ragas-backed metric scoring. Offered as an opt-in alternative to the
default DeepEval backend.

KNOWN ISSUE (verified against ragas==0.4.3, langchain-community==0.4.2,
July 2026): `import ragas` transitively imports
`langchain_community.chat_models.vertexai`, which no longer exists in
current langchain-community releases (langchain-community is being
sunset upstream and no longer ships that module). This means a plain
`pip install ragas` can be import-broken out of the box, independent of
anything in this project.

Workarounds, in order of preference:
  1. Pin an older `langchain-community` known to still ship the vertexai
     chat module, and accept the version conflict risk with other deps.
  2. Use the DeepEval backend in this project instead (default).
  3. Track https://github.com/explodinggradients/ragas for a release
     that drops the vertexai import or makes it lazy, and re-pin.

This backend imports ragas lazily (inside __init__, not at module load)
specifically so that importing `rag_governance_eval` at all doesn't
blow up for users who never asked for the Ragas backend.
"""

from __future__ import annotations

from typing import List, Optional

from .base import MetricBackend, MetricScores, BackendUnavailableError


class RagasBackend(MetricBackend):
    name = "ragas"

    def __init__(self) -> None:
        try:
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                LLMContextRecall,
            )
            from ragas import SingleTurnSample
        except Exception as exc:
            raise BackendUnavailableError(
                "ragas could not be imported. This is frequently caused by a "
                "known ragas 0.4.x issue where `langchain_community.chat_models"
                ".vertexai` no longer exists in current langchain-community "
                "releases. See this module's docstring for workarounds, or "
                "use the DeepEval backend instead. "
                f"Original error: {exc!r}"
            ) from exc

        self._faithfulness = Faithfulness()
        self._context_precision = LLMContextPrecisionWithoutReference()
        self._context_recall = LLMContextRecall()
        self._sample_cls = SingleTurnSample

    def score(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> MetricScores:
        sample = self._sample_cls(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )

        faithfulness_score = self._faithfulness.single_turn_score(sample)
        precision_score = self._context_precision.single_turn_score(sample)
        recall_score = (
            self._context_recall.single_turn_score(sample)
            if reference is not None
            else None
        )

        return MetricScores(
            faithfulness=faithfulness_score,
            hallucination=1.0 - faithfulness_score,
            context_precision=precision_score,
            context_recall=recall_score,
        )
