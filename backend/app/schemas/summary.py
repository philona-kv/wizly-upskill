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


class SummaryUpdatePayload(BaseModel):
    summary_text: str = Field(min_length=1, max_length=8000)
