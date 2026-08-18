from unittest.mock import patch

from rag_governance_eval.metrics.base import MetricBackend, MetricScores
from rag_governance_eval.pipeline import EvalPipeline
from rag_governance_eval.policy.score_actions import PolicyAction


class FakeBackend(MetricBackend):
    """Deterministic stand-in so pipeline tests don't need a judge LLM."""

    name = "fake"

    def __init__(self, canned_scores: MetricScores):
        self.canned_scores = canned_scores

    def score(self, query, answer, contexts, reference=None):
        return self.canned_scores


def _pipeline_with_fake_backend(canned_scores, tmp_path=None):
    fake = FakeBackend(canned_scores)
    with patch("rag_governance_eval.pipeline.get_backend", return_value=fake):
        kwargs = {}
        if tmp_path is not None:
            kwargs["audit_log_path"] = str(tmp_path / "audit.jsonl")
        return EvalPipeline(agent_id="test-agent", backend="fake", **kwargs)


def test_pipeline_passes_high_quality_answer(tmp_path):
    pipeline = _pipeline_with_fake_backend(
        MetricScores(
            faithfulness=0.95,
            hallucination=0.05,
            context_precision=0.9,
            context_recall=0.9,
        ),
        tmp_path,
    )
    result = pipeline.evaluate(
        query="What is our refund policy?",
        answer="Refunds are processed within 5 business days.",
        contexts=["Our refund policy: 5 business days processing time."],
    )
    assert result.action == PolicyAction.PASS
    assert result.audit_record.policy_action == PolicyAction.PASS


def test_pipeline_blocks_low_faithfulness_and_writes_audit(tmp_path):
    pipeline = _pipeline_with_fake_backend(
        MetricScores(
            faithfulness=0.1,
            hallucination=0.9,
            context_precision=0.9,
            context_recall=0.9,
        ),
        tmp_path,
    )
    result = pipeline.evaluate(
        query="What is our refund policy?",
        answer="We offer unlimited free refunds forever with no conditions.",
        contexts=["Our refund policy: 5 business days processing time."],
    )
    assert result.action == PolicyAction.BLOCK
    assert result.triggered_by in ("faithfulness", "hallucination")

    audit_path = tmp_path / "audit.jsonl"
    assert audit_path.exists()
    assert "block" in audit_path.read_text()
