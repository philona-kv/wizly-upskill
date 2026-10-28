from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    summary_text: str
    version: int
    summarized_message_count: int
    updated_at: datetime


class ConversationSummaryState(BaseModel):
    """GET /conversations/{id}/summary — present once generation has run at least once."""

    conversation_id: uuid.UUID
    summary_text: str | None = None
    version: int = 0
    summarized_message_count: int = 0
    updated_at: datetime | None = None


class SummaryUpdatePayload(BaseModel):
    summary_text: str = Field(min_length=1, max_length=8000)
