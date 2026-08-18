from rag_governance_eval.confidence import confidence_label


def test_high_faithfulness_is_well_grounded():
    result = confidence_label(0.9)
    assert result.label == "well-grounded"


def test_mid_faithfulness_is_moderate():
    result = confidence_label(0.5)
    assert result.label == "moderate confidence"


def test_low_faithfulness_is_low_confidence():
    result = confidence_label(0.1)
    assert result.label == "low confidence"


def test_boundary_values():
    assert confidence_label(0.7).label == "well-grounded"
    assert confidence_label(0.6999).label == "moderate confidence"
    assert confidence_label(0.4).label == "moderate confidence"
    assert confidence_label(0.3999).label == "low confidence"


def test_none_is_unknown():
    result = confidence_label(None)
    assert result.label == "unknown"