"""Workflow execution status persistence."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.workflow_execution import WorkflowExecution, WorkflowExecutionStatus


class WorkflowExecutionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, execution_id: uuid.UUID, owner_id: uuid.UUID) -> WorkflowExecution | None:
        stmt = select(WorkflowExecution).where(
            WorkflowExecution.id == execution_id,
            WorkflowExecution.owner_id == owner_id,
        )
        return await self.session.scalar(stmt)

    async def get_completed_by_client_message(
        self, conversation_id: uuid.UUID, client_message_id: uuid.UUID, owner_id: uuid.UUID
    ) -> WorkflowExecution | None:
        stmt = select(WorkflowExecution).where(
            WorkflowExecution.conversation_id == conversation_id,
            WorkflowExecution.client_message_id == client_message_id,
            WorkflowExecution.owner_id == owner_id,
            WorkflowExecution.status == WorkflowExecutionStatus.COMPLETED.value,
        )
        return await self.session.scalar(stmt)

    def add(self, row: WorkflowExecution) -> None:
        self.session.add(row)
