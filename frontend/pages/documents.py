"""Document Management page.

Rerun safety: every side-effecting action (upload, retry, delete) runs inside a widget
callback, which Streamlit executes exactly once per click - never on ordinary reruns.
Uploads are additionally fingerprinted per session, and the backend deduplicates by
SHA-256, so a double-click cannot create duplicate documents.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from components.documents import document_metadata, documents_table, status_label
from components.state import flash, get_api_client, render_flashes
from services.api_client import ApiError

PAGE_SIZE = 20
POLL_INTERVAL_SECONDS = 3
MAX_POLLS = 200  # ~10 minutes of auto-refresh, then manual refresh only

client = get_api_client()
state = st.session_state
state.setdefault("submitted_uploads", {})
state.setdefault("page_offset", 0)
state.setdefault("poll_count", 0)


def _reset_polling() -> None:
    state.poll_count = 0


# -- callbacks (run once per click) ------------------------------------------


def _on_upload() -> None:
    uploaded = state.get("upload_file")
    if uploaded is None:
        flash("warning", "Choose a PDF or image file first.")
        return
    content = uploaded.getvalue()
    visibility = state.get("upload_visibility", "global")
    fingerprint = f"{hashlib.sha256(content).hexdigest()}:{visibility}"
    if fingerprint in state.submitted_uploads:
        flash("info", f"'{uploaded.name}' was already submitted in this session.")
        return
    try:
        result = client.upload_document(
            uploaded.name, content, visibility=visibility, title=state.get("upload_title")
        )
    except ApiError as exc:
        flash("error", f"Upload failed: {exc.message}")
        return
    state.submitted_uploads[fingerprint] = result.document.id
    state.selected_document = result.document.id
    _reset_polling()
    flash("info" if result.duplicate else "success", result.message)


def _on_retry(document_id: str) -> None:
    try:
        client.retry_document(document_id)
    except ApiError as exc:
        flash("error", f"Retry failed: {exc.message}")
        return
    _reset_polling()
    flash("success", "Ingestion re-scheduled.")


def _on_request_delete(document_id: str) -> None:
    state.confirm_delete = document_id


def _on_cancel_delete() -> None:
    state.pop("confirm_delete", None)


def _on_confirm_delete(document_id: str) -> None:
    if state.pop("confirm_delete", None) != document_id:
        return  # stale confirmation; never delete something the user did not confirm
    try:
        client.delete_document(document_id)
    except ApiError as exc:
        if exc.status_code == 404:
            flash("info", "The document was already deleted.")
        else:
            flash("error", f"Delete failed: {exc.message} You can retry the deletion.")
        return
    if state.get("selected_document") == document_id:
        state.pop("selected_document", None)
    flash("success", "Document deleted.")


def _on_refresh() -> None:
    _reset_polling()


def _on_filters_changed() -> None:
    state.page_offset = 0


def _on_page(delta: int) -> None:
    state.page_offset = max(0, state.page_offset + delta)


# -- layout ----------------------------------------------------------------------

st.title("Documents")
st.caption(
    "Upload PDFs or images (PNG, JPG, JPEG) to build the AI Twin's knowledge. "
    "**Global** documents will be available to the AI Twin; **Private** documents "
    "are reserved for the future Playground. Images are transcribed with OCR."
)
render_flashes()

with st.form("upload_form", clear_on_submit=True):
    st.file_uploader("PDF or image file", type=["pdf", "png", "jpg", "jpeg"], key="upload_file")
    st.text_input("Title (optional)", key="upload_title", max_chars=255)
    st.radio(
        "Visibility",
        options=["global", "private"],
        format_func=str.capitalize,
        key="upload_visibility",
        horizontal=True,
    )
    st.form_submit_button("Upload", type="primary", on_click=_on_upload)

st.divider()

header, refresh = st.columns([4, 1])
header.subheader("Your documents")
refresh.button("Refresh", on_click=_on_refresh, width="stretch")

filter_status, filter_visibility = st.columns(2)
status_filter = filter_status.selectbox(
    "Status",
    options=[None, "uploaded", "processing", "ready", "failed", "deleting"],
    format_func=lambda v: "All" if v is None else v.capitalize(),
    key="filter_status",
    on_change=_on_filters_changed,
)
visibility_filter = filter_visibility.selectbox(
    "Visibility",
    options=[None, "global", "private"],
    format_func=lambda v: "All" if v is None else v.capitalize(),
    key="filter_visibility",
    on_change=_on_filters_changed,
)

try:
    page = client.list_documents(
        limit=PAGE_SIZE,
        offset=state.page_offset,
        status=status_filter,
        visibility=visibility_filter,
    )
except ApiError as exc:
    st.error(f"Could not load documents: {exc.message}")
    st.stop()

if not page.items:
    st.info("No documents yet. Upload a PDF above to get started.")
    st.stop()

documents_table(page.items)

prev_col, info_col, next_col = st.columns([1, 3, 1])
prev_col.button("Previous", disabled=page.offset == 0, on_click=_on_page, args=(-PAGE_SIZE,))
info_col.caption(f"Showing {page.offset + 1}-{page.offset + len(page.items)} of {page.total}")
next_col.button(
    "Next",
    disabled=page.offset + len(page.items) >= page.total,
    on_click=_on_page,
    args=(PAGE_SIZE,),
)

# -- status polling ----------------------------------------------------------------

pending = {d.id: d.status for d in page.items if d.status in ("uploaded", "processing")}


def _poll_statuses() -> None:
    state.poll_count += 1
    changed = False
    for document_id, previous in pending.items():
        try:
            current = client.get_status(document_id).status
        except ApiError as exc:
            if exc.status_code == 404:
                changed = True
                continue
            st.caption(f"Status check failed: {exc.message}")
            return
        changed = changed or current != previous
    if changed or state.poll_count >= MAX_POLLS:
        st.rerun(scope="app")  # refresh the full list; the poller stops once nothing is pending
    st.caption(f"{len(pending)} document(s) processing - checking every {POLL_INTERVAL_SECONDS}s.")


if pending and state.poll_count < MAX_POLLS:
    st.fragment(run_every=POLL_INTERVAL_SECONDS)(_poll_statuses)()
elif pending:
    st.caption("Auto-refresh paused. Use **Refresh** to check progress.")

# -- details / actions ---------------------------------------------------------------

st.subheader("Details")
ids = [d.id for d in page.items]
by_id = {d.id: d for d in page.items}
default_index = ids.index(state.selected_document) if state.get("selected_document") in ids else 0
selected_id = st.selectbox(
    "Document",
    options=ids,
    index=default_index,
    format_func=lambda i: f"{by_id[i].title} ({by_id[i].status})",
)
state.selected_document = selected_id
document = by_id[selected_id]

with st.container(border=True):
    document_metadata(document)

    actions = st.columns(4)
    if document.status == "failed":
        actions[0].button("Retry ingestion", on_click=_on_retry, args=(document.id,))
    elif document.status == "uploaded":
        actions[0].button(
            "Re-queue ingestion",
            on_click=_on_retry,
            args=(document.id,),
            help="Use this if a document stays queued, e.g. after a server restart.",
        )
    if state.get("confirm_delete") == document.id:
        st.warning(
            f"Delete **{document.title}**? This removes the file, its chunks, and embeddings."
        )
        confirm, cancel = st.columns(2)
        confirm.button(
            "Confirm delete", type="primary", on_click=_on_confirm_delete, args=(document.id,)
        )
        cancel.button("Cancel", on_click=_on_cancel_delete)
    else:
        label = "Resume deletion" if document.status == "deleting" else "Delete"
        actions[-1].button(label, on_click=_on_request_delete, args=(document.id,))

legend = " | ".join(
    status_label(s) for s in ("uploaded", "processing", "ready", "failed", "deleting")
)
st.caption(f"Status legend: {legend}")
