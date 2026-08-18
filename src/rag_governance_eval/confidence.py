from __future__ import annotations

from typing import NamedTuple


class ConfidenceLabel(NamedTuple):
    label: str
    description: str


def confidence_label(faithfulness: float | None) -> ConfidenceLabel:
    if faithfulness is None:
        return ConfidenceLabel("unknown", "Confidence could not be assessed.")
    if faithfulness >= 0.7:
        return ConfidenceLabel(
            "well-grounded", "This answer is well-supported by the retrieved sources."
        )
    if faithfulness >= 0.4:
        return ConfidenceLabel(
            "moderate confidence",
            "This answer is partially supported by the retrieved sources - verify anything important.",
        )
    return ConfidenceLabel(
        "low confidence",
        "This answer may not be well-supported by the retrieved sources - verify before relying on it.",
    )