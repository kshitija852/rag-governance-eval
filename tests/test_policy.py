from rag_governance_eval.policy.score_actions import ScoreThresholds, PolicyAction, decide_action


def test_high_scores_pass():
    t = ScoreThresholds()
    scores = {"faithfulness": 0.95, "hallucination": 0.05, "context_precision": 0.9, "context_recall": 0.9}
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.PASS
    assert trig is None


def test_uncomputed_precision_recall_do_not_spuriously_flag():
    t = ScoreThresholds()
    scores = {"faithfulness": 0.95, "hallucination": 0.05, "context_precision": None, "context_recall": None}
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.PASS
    assert trig is None


def test_low_answer_relevancy_flags():
    t = ScoreThresholds()
    scores = {
        "faithfulness": 0.95, "hallucination": 0.05,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 0.2,
    }
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.FLAG
    assert trig == "answer_relevancy"


def test_uncomputed_safety_metrics_do_not_spuriously_flag():
    t = ScoreThresholds()
    scores = {
        "faithfulness": 0.95, "hallucination": 0.05,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 0.9,
        "toxicity": None, "bias": None, "pii_leakage": None,
    }
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.PASS


def test_high_toxicity_blocks_when_computed():
    t = ScoreThresholds()
    scores = {
        "faithfulness": 0.95, "hallucination": 0.05,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 0.9,
        "toxicity": 0.8,
        "bias": None, "pii_leakage": None,
    }
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.BLOCK
    assert trig == "toxicity"


def test_low_pii_safety_score_flags_before_blocking():
    """
    pii_leakage uses below-thresholds (higher score = safer, same
    direction as faithfulness) - flag_below=0.7, block_below=0.5.
    A score of 0.6 sits in the flagged-but-not-blocked zone.
    """
    t = ScoreThresholds()
    scores = {
        "faithfulness": 0.95, "hallucination": 0.05,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 0.9,
        "toxicity": 0.1, "bias": 0.1,
        "pii_leakage": 0.6,
    }
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.FLAG
    assert trig == "pii_leakage"


def test_block_takes_priority_over_multiple_flags():
    t = ScoreThresholds()
    scores = {
        "faithfulness": 0.2,
        "hallucination": 0.05,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 0.2,
        "toxicity": None, "bias": None, "pii_leakage": None,
    }
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.BLOCK
    assert trig == "faithfulness"


def test_high_pii_leakage_score_means_safe_not_blocked():
    """
    Regression test for a real bug found live: PIILeakageMetric's score
    direction is the OPPOSITE of ToxicityMetric/BiasMetric in DeepEval's
    own implementation - higher score = SAFER. The original thresholds
    treated it like toxicity/bias, causing a fully safe answer
    (pii_leakage=1.0) to be incorrectly BLOCKED in a live run.
    """
    t = ScoreThresholds()
    scores = {
        "faithfulness": 1.0, "hallucination": 0.1,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 1.0,
        "toxicity": 0.0, "bias": 0.0,
        "pii_leakage": 1.0,
    }
    action, trig = decide_action(scores, t)
    assert trig != "pii_leakage"
    assert action == PolicyAction.PASS


def test_low_pii_leakage_score_means_leaky_and_blocks():
    t = ScoreThresholds()
    scores = {
        "faithfulness": 1.0, "hallucination": 0.1,
        "context_precision": None, "context_recall": None,
        "answer_relevancy": 1.0,
        "toxicity": 0.0, "bias": 0.0,
        "pii_leakage": 0.1,
    }
    action, trig = decide_action(scores, t)
    assert action == PolicyAction.BLOCK
    assert trig == "pii_leakage"