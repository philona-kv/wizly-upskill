"""Owner persistent memories (preferences and facts extracted from chat)."""

from __future__ import annotations

import streamlit as st

from components.state import flash, get_api_client, render_flashes
from services.api_client import ApiError

client = get_api_client()
render_flashes()

st.title("Memories")
st.caption(
    "Facts and preferences the Twin remembered from your conversations. "
    "These are not document citations — deactivate anything incorrect."
)

try:
    data = client.list_memories(limit=100)
except ApiError as exc:
    st.error(f"Could not load memories: {exc.message}")
    st.stop()

items = data.get("items") or []
if not items:
    st.info("No memories stored yet. Keep chatting — extraction runs in the background.")
    st.stop()

for item in items:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"**{item.get('memory_type', 'memory')}** — {item.get('content', '')}")
        st.caption(f"Source: {item.get('source_type', 'unknown')}")
    with col2:
        if st.button("Forget", key=f"forget_{item.get('id')}"):
            try:
                client.deactivate_memory(str(item["id"]))
                flash("success", "Memory deactivated.")
                st.rerun()
            except ApiError as exc:
                flash("error", exc.message)
