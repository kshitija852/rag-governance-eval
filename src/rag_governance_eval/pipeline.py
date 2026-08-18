"""
Orchestrates one governed-RAG-plus-eval call:

    1. adapters.GovernedRAGSource.run(query)   -> answer, contexts   (AGT)
    2. metrics.get_backend(...).score(...)     -> MetricScores       (Ragas/DeepEval)
    3. policy.decide_action(...)               -> PolicyAction       (this project)
    4. audit.EvalAuditRecord.create(...)       -> persisted record   (this project)

Step 1 is out of scope for this module (callers construct a
GovernedRAGSource themselves, since it needs their retriever and answer
function) - this module starts from an already-produced
(query, answer, contexts) triple so it can be used equally well from a
single-turn RAG call or from an agent loop's per-step evaluation hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .metrics import get_backend
from .metrics.base import MetricScores
from .policy.score_actions import ScoreThresholds, PolicyAction, decide_action
from .audit.eval_record import EvalAuditRecord, JsonlAuditWriter


@dataclass
class EvalResult:
    scores: MetricScores
    action: PolicyAction
    triggered_by: Optional[str]
    audit_record: EvalAuditRecord


class EvalPipeline:
    def __init__(
        self,
        *,
        agent_id: str = "rag-governance-eval",
        backend: str = "deepeval",
        thresholds: Optional[ScoreThresholds] = None,
        audit_log_path: Optional[str] = None,
        backend_kwargs: Optional[dict] = None,
    ) -> None:
        self.agent_id = agent_id
        self.backend_name = backend
        self.backend = get_backend(backend, **(backend_kwargs or {}))
        self.thresholds = thresholds or ScoreThresholds()
        self.audit_writer = JsonlAuditWriter(audit_log_path) if audit_log_path else None

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> EvalResult:
        scores = self.backend.score(query, answer, contexts, reference=reference)
        action, triggered_by = decide_action(scores.as_dict(), self.thresholds)

        record = EvalAuditRecord.create(
            agent_id=self.agent_id,
            query=query,
            backend=self.backend_name,
            scores=scores.as_dict(),
            policy_action=action,
            triggered_by=triggered_by,
        )
        if self.audit_writer is not None:
            self.audit_writer.write(record)

        return EvalResult(
            scores=scores, action=action, triggered_by=triggered_by, audit_record=record
        )
