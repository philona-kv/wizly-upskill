from __future__ import annotations

import streamlit as st

from components.state import get_api_client
from services.api_client import ApiError

st.header("Playground — Prompt Experiments")
st.info("Create experiments via the API; this view lists templates for comparison setup.")
client = get_api_client()

try:
    data = client.list_prompt_templates(limit=100)
except ApiError as exc:
    st.error(exc.message)
    st.stop()

for tpl in data.get("items") or []:
    with st.expander(tpl.get("name", "Template")):
        st.caption(tpl.get("description") or "")
        st.write(f"System template: **{tpl.get('is_system')}** · version **{tpl.get('current_version')}**")
        versions = client.list_prompt_versions(tpl["id"])
        for v in versions[:3]:
            st.code(v.get("template_text", "")[:500])
