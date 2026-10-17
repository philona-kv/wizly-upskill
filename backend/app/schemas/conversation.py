from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    citation_id: str
    document_id: uuid.UUID
    document_title: str
    original_filename: str
    page_start: int
    page_end: int
    chunk_id: uuid.UUID
    excerpt: str | None = None


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationRead]
    total: int
    limit: int
    offset: int


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=16_000)
    client_message_id: uuid.UUID | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] | None = None
    model_name: str | None = None
    client_message_id: uuid.UUID | None = None
    created_at: datetime


class MessageList(BaseModel):
    items: list[MessageRead]
    total: int
    limit: int
    offset: int


class ChatTurnResponse(BaseModel):
    conversation_id: uuid.UUID
    user_message: MessageRead
    assistant_message: MessageRead
    operational_metadata: dict[str, Any] | None = None


def citations_from_json(raw: Any) -> list[Citation] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return [Citation.model_validate(item) for item in raw]


def message_read(message) -> MessageRead:
    return MessageRead(
        id=message.id,
        role=message.role,
        content=message.content,
        citations=citations_from_json(message.citations),
        model_name=message.model_name,
        client_message_id=message.client_message_id,
        created_at=message.created_at,
    )
