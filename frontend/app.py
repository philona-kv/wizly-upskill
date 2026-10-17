"""AI Twin Streamlit entry point: ``streamlit run frontend/app.py``.

The UI talks to FastAPI over HTTP only (STREAMLIT_API_URL). It never holds database,
OpenAI, or Supabase credentials.
"""

from __future__ import annotations

import streamlit as st

from components.state import get_api_client
from services.api_client import ApiError

st.set_page_config(page_title="AI Twin", page_icon=":material/psychology:", layout="wide")


@st.cache_data(ttl=15, show_spinner=False)
def backend_status() -> tuple[str, dict]:
    client = get_api_client()
    try:
        health = client.health()
        ready = client.readiness()
    except ApiError as exc:
        return "down", {"message": exc.message}
    return ("ready" if ready.get("status") == "ready" else "degraded"), {**health, **ready}


ai_twin_pages = [
    st.Page(
        "pages/documents.py", title="Documents", icon=":material/description:", default=True
    ),
    st.Page("pages/chat.py", title="AI Twin Chat", icon=":material/chat:"),
    st.Page("pages/memories.py", title="Memories", icon=":material/memory:"),
]

playground_pages = [
    st.Page("pages/playground/library.py", title="Document Library", icon=":material/folder:"),
    st.Page("pages/playground/query_lab.py", title="Query Lab", icon=":material/science:"),
    st.Page(
        "pages/playground/experiments.py", title="Prompt Experiments", icon=":material/tune:"
    ),
    st.Page(
        "pages/playground/evaluation.py",
        title="Evaluation Results",
        icon=":material/analytics:",
    ),
]

pg = st.navigation({"AI Twin": ai_twin_pages, "Playground": playground_pages})

with st.sidebar:
    st.header("AI Twin")
    status, details = backend_status()
    if status == "ready":
        st.success("Backend connected", icon=":material/check_circle:")
    elif status == "degraded":
        missing = [k for k, v in details.get("checks", {}).items() if v != "ok"]
        st.warning(
            "Backend reachable but not fully configured"
            + (f": {', '.join(missing)}" if missing else "."),
            icon=":material/warning:",
        )
    else:
        st.error(details.get("message", "Backend unavailable"), icon=":material/error:")
    if st.button("Re-check backend", width="stretch"):
        backend_status.clear()
        st.rerun()

    st.divider()
    st.caption("Playground: private docs, scoped retrieval, prompt experiments.")
    if details.get("development_identity", True):
        st.warning(
            "**Development mode.** All requests use a single development identity. "
            "This is not secure multi-user software - do not expose it publicly.",
            icon=":material/lock_open:",
        )

pg.run()
