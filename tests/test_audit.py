import json

from rag_governance_eval.audit.eval_record import EvalAuditRecord, JsonlAuditWriter
from rag_governance_eval.policy.score_actions import PolicyAction


def test_query_text_never_persisted():
    record = EvalAuditRecord.create(
        agent_id="agent-1",
        query="what is the CEO's home address",  # sensitive-looking query
        backend="deepeval",
        scores={"faithfulness": 0.9},
        policy_action=PolicyAction.PASS,
    )
    dumped = record.to_json()
    assert "home address" not in dumped
    assert "CEO" not in dumped
    assert len(record.query_hash) == 64  # sha256 hex digest length


def test_jsonl_writer_appends_valid_json_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    writer = JsonlAuditWriter(path)

    r1 = EvalAuditRecord.create(
        agent_id="agent-1",
        query="q1",
        backend="deepeval",
        scores={"faithfulness": 0.9},
        policy_action=PolicyAction.PASS,
    )
    r2 = EvalAuditRecord.create(
        agent_id="agent-1",
        query="q2",
        backend="deepeval",
        scores={"faithfulness": 0.2},
        policy_action=PolicyAction.BLOCK,
        triggered_by="faithfulness",
    )
    writer.write(r1)
    writer.write(r2)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["policy_action"] == "pass"
    assert parsed[1]["policy_action"] == "block"
    assert parsed[1]["triggered_by"] == "faithfulness"
