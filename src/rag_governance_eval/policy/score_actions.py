from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class PolicyAction(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"
    RETRY = "retry"


class ScoreThresholds(BaseModel):
    faithfulness_flag_below: float = 0.7
    faithfulness_block_below: Optional[float] = 0.4
    hallucination_flag_above: float = 0.3
    hallucination_block_above: Optional[float] = 0.6
    context_precision_flag_below: float = 0.5
    context_precision_block_below: Optional[float] = None
    context_recall_flag_below: float = 0.5
    context_recall_block_below: Optional[float] = None
    answer_relevancy_flag_below: float = 0.5
    answer_relevancy_block_below: Optional[float] = None
    toxicity_flag_above: float = 0.5
    toxicity_block_above: Optional[float] = 0.7
    bias_flag_above: float = 0.5
    bias_block_above: Optional[float] = None
    # PIILeakageMetric's score direction is the OPPOSITE of
    # ToxicityMetric/BiasMetric in DeepEval - higher score = SAFER
    # (more verdicts found no leakage), same direction as faithfulness.
    pii_leakage_flag_below: float = 0.7
    pii_leakage_block_below: Optional[float] = 0.5


def decide_action(scores: dict, thresholds: ScoreThresholds):
    checks = [
        ("faithfulness", scores.get("faithfulness"), thresholds.faithfulness_block_below, "below"),
        ("faithfulness", scores.get("faithfulness"), thresholds.faithfulness_flag_below, "below"),
        ("hallucination", scores.get("hallucination"), thresholds.hallucination_block_above, "above"),
        ("hallucination", scores.get("hallucination"), thresholds.hallucination_flag_above, "above"),
        ("context_precision", scores.get("context_precision"), thresholds.context_precision_block_below, "below"),
        ("context_precision", scores.get("context_precision"), thresholds.context_precision_flag_below, "below"),
        ("context_recall", scores.get("context_recall"), thresholds.context_recall_block_below, "below"),
        ("context_recall", scores.get("context_recall"), thresholds.context_recall_flag_below, "below"),
        ("answer_relevancy", scores.get("answer_relevancy"), thresholds.answer_relevancy_block_below, "below"),
        ("answer_relevancy", scores.get("answer_relevancy"), thresholds.answer_relevancy_flag_below, "below"),
        ("toxicity", scores.get("toxicity"), thresholds.toxicity_block_above, "above"),
        ("toxicity", scores.get("toxicity"), thresholds.toxicity_flag_above, "above"),
        ("bias", scores.get("bias"), thresholds.bias_block_above, "above"),
        ("bias", scores.get("bias"), thresholds.bias_flag_above, "above"),
        ("pii_leakage", scores.get("pii_leakage"), thresholds.pii_leakage_block_below, "below"),
        ("pii_leakage", scores.get("pii_leakage"), thresholds.pii_leakage_flag_below, "below"),
    ]

    def crosses(value, bound, direction):
        if value is None or bound is None:
            return False
        return value < bound if direction == "below" else value > bound

    for i in range(0, len(checks), 2):
        name, value, bound, direction = checks[i]
        if crosses(value, bound, direction):
            return PolicyAction.BLOCK, name

    for i in range(1, len(checks), 2):
        name, value, bound, direction = checks[i]
        if crosses(value, bound, direction):
            return PolicyAction.FLAG, name

    return PolicyAction.PASS, None