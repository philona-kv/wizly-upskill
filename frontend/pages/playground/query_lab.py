from __future__ import annotations

import streamlit as st

from components.state import get_api_client
from services.api_client import ApiError

st.header("Playground — Query Lab")
client = get_api_client()

scope = st.sidebar.selectbox(
    "Document scope",
    ["combined", "global_only", "private_only", "selected"],
    format_func=lambda s: s.replace("_", " ").title(),
)
selected_ids = list(st.session_state.get("pg_selected_docs") or [])
if scope == "selected":
    st.sidebar.caption(f"{len(selected_ids)} document(s) selected in Document Library.")

prompts = client.list_prompt_templates()
prompt_items = prompts.get("items") or []
prompt_labels = ["(default Twin instructions)"] + [
    f"{p['name']} (v{p['current_version']})" for p in prompt_items
]
prompt_choice = st.sidebar.selectbox("Prompt template", prompt_labels)
prompt_version_id = None
if prompt_choice != prompt_labels[0] and prompt_items:
    idx = prompt_labels.index(prompt_choice) - 1
    tpl = prompt_items[idx]
    versions = client.list_prompt_versions(tpl["id"])
    if versions:
        prompt_version_id = versions[0]["id"]

question = st.text_area("Question", height=120)
if st.button("Run query", type="primary"):
    payload = {
        "question": question,
        "scope": scope,
        "document_ids": selected_ids if scope == "selected" else [],
    }
    if prompt_version_id:
        payload["prompt_version_id"] = prompt_version_id
    try:
        result = client.playground_query(payload)
        st.session_state.pg_last_result = result
    except ApiError as exc:
        st.error(exc.message)

if result := st.session_state.get("pg_last_result"):
    st.subheader("Answer")
    st.markdown(result.get("answer") or "")
    st.subheader("Citations")
    for c in result.get("citations") or []:
        st.markdown(f"- **{c.get('document_title')}** ({c.get('original_filename')}) — pages {c.get('page_start')}-{c.get('page_end')}")
    with st.expander("Retrieved evidence"):
        for ev in result.get("document_evidence") or []:
            st.markdown(f"**{ev.get('document_title')}** score={ev.get('retrieval_score')}")
            st.caption(ev.get("excerpt") or "")
    with st.expander("Debug metadata"):
        st.json(result.get("operational_metadata") or {})
        st.json(result.get("token_usage") or {})
