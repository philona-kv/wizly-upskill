from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class WorkflowExecutionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    status: str
    current_node: str | None
    attempt_count: int
    graph_version: str
    last_error_code: str | None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None
