"""
Verifies the reference-answer mechanism actually flows through the
whole pipeline - this is what turns permanent None context_precision/
context_recall into real, computed numbers.
"""

from rag_governance_eval.metrics.base import MetricBackend, MetricScores
from rag_governance_eval.pipeline import EvalPipeline
from rag_governance_eval.policy.score_actions import ScoreThresholds


class ReferenceAwareFakeBackend(MetricBackend):
    name = "fake"

    def score(self, query, answer, contexts, reference=None):
        got_reference = reference is not None
        return MetricScores(
            faithfulness=0.9, hallucination=0.1,
            context_precision=0.85 if got_reference else None,
            context_recall=0.8 if got_reference else None,
        )


def make_pipeline():
    pipeline = EvalPipeline.__new__(EvalPipeline)
    pipeline.agent_id = "test"
    pipeline.backend_name = "fake"
    pipeline.backend = ReferenceAwareFakeBackend()
    pipeline.thresholds = ScoreThresholds()
    pipeline.audit_writer = None
    return pipeline


def test_no_reference_keeps_precision_recall_none():
    pipeline = make_pipeline()
    result = pipeline.evaluate(query="q", answer="a", contexts=["c"])
    assert result.scores.context_precision is None
    assert result.scores.context_recall is None


def test_reference_supplied_produces_real_precision_recall():
    pipeline = make_pipeline()
    result = pipeline.evaluate(query="q", answer="a", contexts=["c"], reference="the gold answer")
    assert result.scores.context_precision == 0.85
    assert result.scores.context_recall == 0.8


def test_reference_qa_dataset_loads_and_has_required_shape():
    import importlib.util
    from pathlib import Path

    ref_path = Path(__file__).parent.parent / "examples" / "reference_qa.py"
    spec = importlib.util.spec_from_file_location("reference_qa", ref_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    REFERENCE_QA = module.REFERENCE_QA

    assert len(REFERENCE_QA) >= 1
    for item in REFERENCE_QA:
        assert "question" in item and "reference" in item
        assert isinstance(item["question"], str) and len(item["question"]) > 0
        assert isinstance(item["reference"], str) and len(item["reference"]) > 0