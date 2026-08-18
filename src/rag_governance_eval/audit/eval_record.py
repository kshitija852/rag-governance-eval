"""
Structured, versioned audit records for evaluation outcomes.

Deliberately a separate, explicit schema (not just a dict) so it can be
validated, versioned, and queried - the same reasoning agent-rag-governance
applies to its own RAGAuditEntry, extended to cover output-quality
evaluation rather than access-control decisions.

These records are designed to sit *next to* AGT's own audit log
(agent_rag_governance.audit.AuditLogger / RAGAuditEntry) - correlated by
timestamp/query_hash - rather than replacing it. A single EU AI Act
audit trail is the concatenation of both: "was this access allowed" +
"was this output trustworthy."
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..policy.score_actions import PolicyAction


class EvalAuditRecord(BaseModel):
    schema_version: str = "1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str
    query_hash: str
    backend: str
    scores: dict[str, Optional[float]]
    policy_action: PolicyAction
    triggered_by: Optional[str] = None  # which metric caused the action, if any

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        query: str,
        backend: str,
        scores: dict[str, Optional[float]],
        policy_action: PolicyAction,
        triggered_by: Optional[str] = None,
    ) -> "EvalAuditRecord":
        # Same reasoning as AGT's own audit log: never persist raw query
        # text, only a hash, to avoid leaking sensitive search terms into
        # a compliance artifact.
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return cls(
            agent_id=agent_id,
            query_hash=query_hash,
            backend=backend,
            scores=scores,
            policy_action=policy_action,
            triggered_by=triggered_by,
        )

    def to_json(self) -> str:
        return self.model_dump_json()


class JsonlAuditWriter:
    """Appends one EvalAuditRecord per line to a file, mirroring AGT's
    own JSON-lines audit log format so both can be ingested by the same
    downstream log pipeline."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: EvalAuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_json())
            f.write("\n")