from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    memory_type: str
    content: str
    source_type: str
    source_conversation_id: uuid.UUID | None
    confidence: float
    importance: float
    status: str
    created_at: datetime
    updated_at: datetime


class MemoryList(BaseModel):
    items: list[MemoryRead]
    total: int


class ProposedMemory(BaseModel):
    memory_type: Literal[
        "user_preference",
        "user_provided_fact",
        "project_context",
        "ongoing_goal",
        "decision",
        "unresolved_question",
        "other",
    ]
    content: str = Field(min_length=1, max_length=2000)
    source_message_ids: list[uuid.UUID] = Field(default_factory=list)
    source_type: Literal["user_statement", "assistant_inference"] = "user_statement"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    supersedes_memory_id: uuid.UUID | None = None
    expires_at: datetime | None = None


class MemoryExtractionResult(BaseModel):
    memories: list[ProposedMemory] = Field(default_factory=list)
