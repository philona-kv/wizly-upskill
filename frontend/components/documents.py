"""Presentation helpers for documents (no API calls)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from services.api_client import Document

STATUS_LABELS = {
    "uploaded": ":blue[Queued]",
    "processing": ":orange[Processing]",
    "ready": ":green[Ready]",
    "failed": ":red[Failed]",
    "deleting": ":gray[Deleting]",
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def _fmt_dt(value: datetime | None) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M") if value else "-"


def _fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def documents_table(documents: list[Document]) -> None:
    rows = [
        {
            "Title": d.title,
            "Filename": d.original_filename,
            "Visibility": d.visibility.capitalize(),
            "Status": d.status,
            "Pages": d.page_count,
            "Chunks": d.chunk_count if d.status == "ready" else None,
            "Uploaded": _fmt_dt(d.created_at),
        }
        for d in documents
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def document_metadata(document: Document) -> None:
    st.markdown(f"**Status:** {status_label(document.status)}")
    if document.status == "failed" and document.error_message:
        st.error(document.error_message)
    left, right = st.columns(2)
    left.markdown(
        f"**Filename:** {document.original_filename}  \n"
        f"**Visibility:** {document.visibility.capitalize()}  \n"
        f"**Size:** {_fmt_size(document.file_size_bytes)}  \n"
        f"**Document ID:** `{document.id}`"
    )
    right.markdown(
        f"**Pages:** {document.page_count if document.page_count is not None else '-'}  \n"
        f"**Chunks:** {document.chunk_count}  \n"
        f"**Embedding model:** {document.embedding_model or '-'}  \n"
        f"**Attempts:** {document.ingestion_attempts}"
    )
    st.caption(
        f"Uploaded {_fmt_dt(document.created_at)} | Updated {_fmt_dt(document.updated_at)} | "
        f"Processed {_fmt_dt(document.processed_at)}"
    )
