"""
Implements the actual RETRY behavior - previously PolicyAction.RETRY
existed as an enum value with nothing acting on it. This is what makes
"retry with different retrieval" real: on a flagged/blocked first
attempt, automatically re-retrieve with more context (a larger k) and
re-generate, before anything is returned to the caller.

This is deliberately NOT folded into EvalPipeline.evaluate() itself,
because retrying needs to re-run retrieval + generation
(GovernedRAGSource.run), which the pipeline alone has no access to -
keeping this separate keeps EvalPipeline's job narrow (score what you
give it) and this module's job narrow (orchestrate a second attempt
when the first one is bad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .adapters.agt_adapter import GovernedRAGSource, RetrievedContext
from .pipeline import EvalPipeline, EvalResult
from .policy.score_actions import PolicyAction


@dataclass
class RetryOutcome:
    answer: str
    contexts: List[RetrievedContext]
    result: EvalResult
    attempts: int
    retried: bool
    improved: bool  # True only if a retry happened AND the retry scored better


def evaluate_with_retry(
    source: GovernedRAGSource,
    pipeline: EvalPipeline,
    query: str,
    *,
    retry_k: int = 8,
    retry_on: tuple = (PolicyAction.FLAG, PolicyAction.BLOCK),
) -> RetryOutcome:
    """
    Runs the query once. If the result's action is in `retry_on`,
    automatically runs a second attempt with a larger `k` (more
    retrieved chunks -> a genuinely different retrieval, not just a
    re-roll of the same generation). Keeps whichever attempt has the
    higher faithfulness score.

    A retry is not a guarantee of improvement - if the second attempt
    doesn't score higher, the FIRST attempt's answer is kept (more
    context isn't always better; it can dilute relevance). `improved`
    tells the caller which happened, for logging/debugging.
    """
    answer, contexts = source.run(query)
    result = pipeline.evaluate(query=query, answer=answer, contexts=[c.text for c in contexts])

    if result.action not in retry_on:
        return RetryOutcome(answer, contexts, result, attempts=1, retried=False, improved=False)

    try:
        retry_answer, retry_contexts = source.run(query, k=retry_k)
        retry_result = pipeline.evaluate(
            query=query, answer=retry_answer, contexts=[c.text for c in retry_contexts]
        )
    except Exception:
        # A retry is a best-effort improvement attempt, not a
        # guarantee - if it fails outright (e.g. the judge call times
        # out on a longer context, which is a real, observed failure
        # mode on CPU-only local models), fall back to the original,
        # already-valid first attempt rather than losing the whole
        # request. The user gets an answer either way.
        return RetryOutcome(answer, contexts, result, attempts=2, retried=True, improved=False)

    if retry_result.scores.faithfulness > result.scores.faithfulness:
        return RetryOutcome(
            retry_answer, retry_contexts, retry_result, attempts=2, retried=True, improved=True
        )
    return RetryOutcome(answer, contexts, result, attempts=2, retried=True, improved=False)