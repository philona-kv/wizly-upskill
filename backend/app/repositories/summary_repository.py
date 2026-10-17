"""Conversation summary persistence."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.summary import ConversationSummary


class SummaryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_for_conversation(
        self, conversation_id: uuid.UUID, owner_id: uuid.UUID
    ) -> ConversationSummary | None:
        stmt = select(ConversationSummary).where(
            ConversationSummary.conversation_id == conversation_id,
            ConversationSummary.owner_id == owner_id,
        )
        return await self.session.scalar(stmt)

    def add(self, summary: ConversationSummary) -> None:
        self.session.add(summary)

    async def update_if_version(
        self,
        summary_id: uuid.UUID,
        owner_id: uuid.UUID,
        expected_version: int,
        *,
        summary_text: str,
        source_message_start_id: uuid.UUID | None,
        source_message_end_id: uuid.UUID | None,
        summarized_message_count: int,
    ) -> bool:
        result = await self.session.execute(
            update(ConversationSummary)
            .where(
                ConversationSummary.id == summary_id,
                ConversationSummary.owner_id == owner_id,
                ConversationSummary.version == expected_version,
            )
            .values(
                summary_text=summary_text,
                source_message_start_id=source_message_start_id,
                source_message_end_id=source_message_end_id,
                summarized_message_count=summarized_message_count,
                version=expected_version + 1,
            )
        )
        return result.rowcount > 0
