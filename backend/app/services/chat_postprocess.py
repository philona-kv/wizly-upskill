"""Background memory extraction and conversation summarization after chat turns."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.conversation_repository import ConversationRepository
from app.services.conversation_summary_service import ConversationSummaryService
from app.services.memory_extraction_service import MemoryExtractionService

logger = logging.getLogger(__name__)


class ChatPostProcessService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        extraction: MemoryExtractionService,
        summary: ConversationSummaryService,
        *,
        memory_interval: int,
        summary_trigger: int,
    ) -> None:
        self._session_factory = session_factory
        self._extraction = extraction
        self._summary = summary
        self._memory_interval = memory_interval
        self._summary_trigger = summary_trigger
        self._schedule_fn: Callable[[uuid.UUID, uuid.UUID], None] | None = None

    def set_scheduler(self, fn: Callable[[uuid.UUID, uuid.UUID], None]) -> None:
        self._schedule_fn = fn

    def schedule(self, owner_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        if self._schedule_fn is not None:
            self._schedule_fn(owner_id, conversation_id)

    async def run(self, owner_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        try:
            async with self._session_factory() as session:
                repo = ConversationRepository(session)
                _, total = await repo.list_messages(conversation_id, limit=1, offset=0)
                recent = await repo.recent_messages(conversation_id, limit=80)
            await self._summary.maybe_update(
                owner_id, conversation_id, total, self._summary_trigger
            )
            if total > 0 and total % self._memory_interval == 0:
                await self._extraction.extract_and_store(owner_id, conversation_id, recent)
        except Exception as exc:
            logger.warning(
                "chat_postprocess_failed",
                extra={"conversation_id": str(conversation_id), "exc_type": type(exc).__name__},
            )
