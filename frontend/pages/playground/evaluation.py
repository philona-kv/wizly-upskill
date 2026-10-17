from __future__ import annotations

import streamlit as st

from components.state import get_api_client
from services.api_client import ApiError

st.header("Playground — Evaluation Results")
client = get_api_client()

dataset = st.text_input("Dataset path", value="evaluation/datasets/core.jsonl")
limit = st.number_input("Case limit", min_value=1, max_value=100, value=5)
scope = st.selectbox("Scope", ["global_only", "combined", "private_only"])

if st.button("Run evaluation"):
    try:
        report = client.run_playground_evaluation(
            {"dataset_path": dataset, "limit": int(limit), "scope": scope}
        )
        st.session_state.pg_eval_report = report
    except ApiError as exc:
        st.error(exc.message)

if report := st.session_state.get("pg_eval_report"):
    st.subheader("Aggregate metrics")
    st.json(report.get("aggregate_metrics") or {})
    st.subheader("Per-case results")
    for row in report.get("results") or []:
        with st.expander(row.get("case_id", "case")):
            st.json(row)
