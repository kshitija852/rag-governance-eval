from agent_rag_governance import RAGPolicy

from rag_governance_eval.adapters.agt_adapter import GovernedRAGSource
from rag_governance_eval.metrics.base import MetricBackend, MetricScores
from rag_governance_eval.pipeline import EvalPipeline
from rag_governance_eval.policy.score_actions import ScoreThresholds
from rag_governance_eval.retry import evaluate_with_retry


class KAwareRetriever:
    def invoke(self, query: str, k: int = 4):
        return [{"text": f"chunk {i} for {query}"} for i in range(k)]


def make_pipeline(faithfulness_by_num_contexts: dict) -> EvalPipeline:
    class ContextCountBackend(MetricBackend):
        name = "fake"

        def score(self, query, answer, contexts, reference=None):
            f = faithfulness_by_num_contexts.get(len(contexts), 0.5)
            return MetricScores(
                faithfulness=f, hallucination=1 - f, context_precision=None, context_recall=None
            )

    pipeline = EvalPipeline.__new__(EvalPipeline)
    pipeline.agent_id = "test"
    pipeline.backend_name = "fake"
    pipeline.backend = ContextCountBackend()
    pipeline.thresholds = ScoreThresholds()
    pipeline.audit_writer = None
    return pipeline


def make_source(answer_fn=None):
    return GovernedRAGSource(
        retriever=KAwareRetriever(),
        answer_fn=answer_fn or (lambda q, ctxs: f"answer with {len(ctxs)} contexts"),
        policy=RAGPolicy(audit_enabled=False),
        collection="public_docs",
    )


def test_no_retry_when_first_attempt_passes():
    source = make_source()
    pipeline = make_pipeline({4: 0.95})
    outcome = evaluate_with_retry(source, pipeline, "test query")
    assert outcome.retried is False
    assert outcome.attempts == 1
    assert outcome.result.action.value == "pass"


def test_retry_triggers_and_improves():
    source = make_source()
    pipeline = make_pipeline({4: 0.2, 8: 0.9})
    outcome = evaluate_with_retry(source, pipeline, "test query", retry_k=8)
    assert outcome.retried is True
    assert outcome.improved is True
    assert outcome.attempts == 2
    assert "8 contexts" in outcome.answer


def test_retry_triggers_but_does_not_improve_keeps_original():
    source = make_source()
    pipeline = make_pipeline({4: 0.2, 8: 0.15})
    outcome = evaluate_with_retry(source, pipeline, "test query", retry_k=8)
    assert outcome.retried is True
    assert outcome.improved is False
    assert outcome.attempts == 2
    assert "4 contexts" in outcome.answer