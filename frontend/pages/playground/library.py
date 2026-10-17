from __future__ import annotations

import streamlit as st

from components.state import flash, get_api_client, render_flashes
from services.api_client import ApiError

st.header("Playground — Document Library")
st.caption("Uploads are **private** by default. Promote ready documents to global for AI Twin access.")

render_flashes()
client = get_api_client()

if "pg_selected_docs" not in st.session_state:
    st.session_state.pg_selected_docs = set()

with st.expander("Upload private PDF", expanded=False):
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    title = st.text_input("Title (optional)")
    if st.button("Upload", disabled=uploaded is None):
        try:
            result = client.upload_playground_document(
                uploaded.name, uploaded.getvalue(), title=title or None
            )
            flash("success", result.message)
            st.rerun()
        except ApiError as exc:
            st.error(exc.message)

col1, col2, col3 = st.columns(3)
with col1:
    vis_filter = st.selectbox("Visibility", ["", "private", "global"])
with col2:
    status_filter = st.selectbox("Status", ["", "ready", "processing", "failed", "uploaded"])
with col3:
    search = st.text_input("Search title/filename")

try:
    page = client.list_playground_documents(
        limit=50,
        visibility=vis_filter or None,
        status=status_filter or None,
        search=search or None,
    )
except ApiError as exc:
    st.error(exc.message)
    st.stop()

for doc in page.items:
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"**{doc.title}** (`{doc.original_filename}`)")
            st.caption(
                f"{doc.visibility} · {doc.status} · chunks {doc.chunk_count} · "
                f"pages {doc.page_count or '—'}"
            )
            if doc.error_message:
                st.warning(doc.error_message[:300])
        with c2:
            selected = doc.id in st.session_state.pg_selected_docs
            if st.checkbox("Select for query", value=selected, key=f"sel_{doc.id}"):
                st.session_state.pg_selected_docs.add(doc.id)
            else:
                st.session_state.pg_selected_docs.discard(doc.id)
        with c3:
            if doc.visibility == "private" and doc.status == "ready":
                if st.button("Promote to global", key=f"promote_{doc.id}"):
                    st.session_state[f"confirm_promote_{doc.id}"] = True
            if st.session_state.get(f"confirm_promote_{doc.id}"):
                st.warning("This document will become globally accessible for your AI Twin.")
                if st.button("Confirm promotion", key=f"confirm_yes_{doc.id}"):
                    try:
                        client.promote_playground_document(doc.id)
                        flash("success", "Document promoted to global.")
                        st.session_state.pop(f"confirm_promote_{doc.id}", None)
                        st.rerun()
                    except ApiError as exc:
                        st.error(exc.message)
