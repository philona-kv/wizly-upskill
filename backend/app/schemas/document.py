from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import DocumentStatus, DocumentVisibility


class DocumentRead(BaseModel):
    """Document metadata. Never includes chunk content, embeddings, or storage paths."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    file_hash: str
    visibility: DocumentVisibility
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    chunk_count: int
    embedding_model: str | None
    ingestion_attempts: int
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None


class DocumentList(BaseModel):
    items: list[DocumentRead]
    total: int
    limit: int
    offset: int


class DocumentStatusRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: DocumentStatus
    error_message: str | None
    page_count: int | None
    chunk_count: int
    ingestion_attempts: int
    updated_at: datetime
    processed_at: datetime | None


class DocumentAccepted(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    message: str
    duplicate: bool = False
    document: DocumentRead
