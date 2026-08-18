from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

AUDIT_LOG_PATH = "rag_eval_audit.jsonl"


def load_audit_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp", "agent_id", "backend", "policy_action",
                "triggered_by", "faithfulness", "hallucination",
                "context_precision", "context_recall",
            ]
        )

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scores = row.pop("scores", {})
            row.update(scores)
            records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df


def render_dashboard(df: pd.DataFrame) -> None:
    """Separated from load_audit_log's file I/O so the rendering logic
    can also be exercised in isolation with a hand-built DataFrame."""
    st.set_page_config(page_title="RAG Eval Audit Dashboard", layout="wide")
    st.title("RAG Governance + Eval Audit Dashboard")

    if df.empty:
        st.info(
            f"No audit records found yet. Run a query through "
            f"query_pdfs.py, query_pdfs_langchain.py, or the API "
            f"first - each writes to `{AUDIT_LOG_PATH}`."
        )
        return

    # --- summary metrics row ---
    total = len(df)
    pass_count = (df["policy_action"] == "pass").sum()
    flag_count = (df["policy_action"] == "flag").sum()
    block_count = (df["policy_action"] == "block").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total evaluations", total)
    c2.metric("Passed", int(pass_count))
    c3.metric("Flagged", int(flag_count))
    c4.metric("Blocked", int(block_count))

    st.divider()

    # --- scores over time ---
    st.subheader("Scores over time")
    score_cols = ["faithfulness", "hallucination", "context_precision", "context_recall"]
    available_cols = [c for c in score_cols if c in df.columns]
    if available_cols:
        chart_df = df.set_index("timestamp")[available_cols]
        st.line_chart(chart_df)
    else:
        st.write("No score columns found in the log.")

    st.divider()

    # --- action breakdown + what triggers flags/blocks most ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Actions")
        st.bar_chart(df["policy_action"].value_counts())
    with col_b:
        st.subheader("What triggers flag/block most often")
        triggered = df[df["triggered_by"].notna()]["triggered_by"].value_counts()
        if not triggered.empty:
            st.bar_chart(triggered)
        else:
            st.write("Nothing has been flagged or blocked yet.")

    st.divider()

    # --- filterable raw table ---
    st.subheader("Raw records")
    action_filter = st.multiselect(
        "Filter by action", options=sorted(df["policy_action"].unique()),
        default=list(df["policy_action"].unique()),
    )
    filtered = df[df["policy_action"].isin(action_filter)]
    st.dataframe(
        filtered[
            ["timestamp", "backend", "policy_action", "triggered_by", *available_cols]
        ].sort_values("timestamp", ascending=False),
        use_container_width=True,
    )


def main():
    df = load_audit_log(AUDIT_LOG_PATH)
    render_dashboard(df)


if __name__ == "__main__":
    main()