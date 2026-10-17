from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.models.document_chunk import EMBEDDING_DIMENSIONS


class MemoryType(StrEnum):
    USER_PREFERENCE = "user_preference"
    USER_PROVIDED_FACT = "user_provided_fact"
    PROJECT_CONTEXT = "project_context"
    ONGOING_GOAL = "ongoing_goal"
    DECISION = "decision"
    UNRESOLVED_QUESTION = "unresolved_question"
    CONVERSATION_SUMMARY = "conversation_summary"
    OTHER = "other"


class MemorySourceType(StrEnum):
    USER_STATEMENT = "user_statement"
    ASSISTANT_INFERENCE = "assistant_inference"
    SYSTEM_SUMMARY = "system_summary"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class OwnerMemory(Base):
    __tablename__ = "owner_memories"
    __table_args__ = (
        CheckConstraint(
            "memory_type IN ("
            "'user_preference', 'user_provided_fact', 'project_context', "
            "'ongoing_goal', 'decision', 'unresolved_question', 'conversation_summary', 'other'"
            ")",
            name="owner_memories_type_valid",
        ),
        CheckConstraint(
            "source_type IN ('user_statement', 'assistant_inference', 'system_summary')",
            name="owner_memories_source_type_valid",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'deleted')",
            name="owner_memories_status_valid",
        ),
        Index("ix_owner_memories_owner_id_status", "owner_id", "status"),
        Index(
            "ix_owner_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_content: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    source_message_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=MemoryStatus.ACTIVE.value)
    supersedes_memory_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("owner_memories.id", ondelete="SET NULL"), nullable=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
