"""Durable workflow execution tracking for recovery and idempotency."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models.workflow_execution import WorkflowExecution, WorkflowExecutionStatus
from app.repositories.workflow_execution_repository import WorkflowExecutionRepository
from app.schemas.workflow import WorkflowExecutionRead


def make_thread_id(owner_id: uuid.UUID, conversation_id: uuid.UUID) -> str:
    return f"{owner_id}:{conversation_id}"


class WorkflowExecutionService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def start(
        self,
        execution_id: uuid.UUID,
        *,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
        client_message_id: uuid.UUID | None,
        graph_version: str,
    ) -> None:
        row = WorkflowExecution(
            id=execution_id,
            owner_id=owner_id,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            client_message_id=client_message_id,
            thread_id=make_thread_id(owner_id, conversation_id),
            graph_version=graph_version,
            status=WorkflowExecutionStatus.RUNNING.value,
        )
        async with self._session_factory() as session, session.begin():
            WorkflowExecutionRepository(session).add(row)

    async def complete(self, execution_id: uuid.UUID, owner_id: uuid.UUID) -> None:
        async with self._session_factory() as session, session.begin():
            row = await WorkflowExecutionRepository(session).get(execution_id, owner_id)
            if row is None:
                return
            row.status = WorkflowExecutionStatus.COMPLETED.value
            row.completed_at = datetime.now(UTC)

    async def fail(self, execution_id: uuid.UUID, owner_id: uuid.UUID, error_code: str) -> None:
        async with self._session_factory() as session, session.begin():
            row = await WorkflowExecutionRepository(session).get(execution_id, owner_id)
            if row is None:
                return
            row.status = WorkflowExecutionStatus.FAILED.value
            row.last_error_code = error_code
            row.completed_at = datetime.now(UTC)

    async def get(
        self, execution_id: uuid.UUID, owner_id: uuid.UUID
    ) -> WorkflowExecutionRead | None:
        async with self._session_factory() as session:
            row = await WorkflowExecutionRepository(session).get(execution_id, owner_id)
            if row is None:
                return None
            return WorkflowExecutionRead.model_validate(row)

    async def find_completed_client(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        client_message_id: uuid.UUID,
    ) -> WorkflowExecution | None:
        async with self._session_factory() as session:
            return await WorkflowExecutionRepository(session).get_completed_by_client_message(
                conversation_id, client_message_id, owner_id
            )
