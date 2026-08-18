"""
rag-governance-eval
~~~~~~~~~~~~~~~~~~~

Evaluation and quality-gating layer that sits alongside Microsoft's
Agent Governance Toolkit (the `agent-rag-governance` package).

This package does NOT modify or vendor AGT's source. It imports
`agent-rag-governance` as a normal dependency and adds a layer on top:

    RAG Application
          |
    agent-rag-governance   (access control, rate limiting, PII scan, audit)
          |
    rag-governance-eval     (THIS PACKAGE: faithfulness / hallucination /
          |                  context precision & recall, scored via Ragas
          |                  or DeepEval, with policy actions on low scores)
          v
    Metrics + audit records
"""

from .pipeline import EvalPipeline, EvalResult
from .policy.score_actions import ScoreThresholds, PolicyAction
from .audit.eval_record import EvalAuditRecord

__version__ = "0.1.0"

__all__ = [
    "EvalPipeline",
    "EvalResult",
    "ScoreThresholds",
    "PolicyAction",
    "EvalAuditRecord",
]
