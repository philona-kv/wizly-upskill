"""Persistence for conversations and messages. Callers own transaction boundaries."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message, MessageRole


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_for_owner(
        self, conversation_id: uuid.UUID, owner_id: uuid.UUID
    ) -> Conversation | None:
        stmt = select(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == owner_id
        )
        return await self.session.scalar(stmt)

    async def list_for_owner(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Conversation], int]:
        conditions = [Conversation.owner_id == owner_id]
        total = (
            await self.session.scalar(
                select(func.count()).select_from(Conversation).where(*conditions)
            )
            or 0
        )
        stmt = (
            select(Conversation)
            .where(*conditions)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.scalars(stmt)).all()), total

    def add(self, conversation: Conversation) -> None:
        self.session.add(conversation)

    async def delete(self, conversation_id: uuid.UUID, owner_id: uuid.UUID) -> bool:
        result = await self.session.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id, Conversation.owner_id == owner_id
            )
        )
        return result.rowcount > 0

    async def touch(self, conversation_id: uuid.UUID) -> None:
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=func.now())
            .execution_options(synchronize_session=False)
        )

    async def set_title_if_default(self, conversation_id: uuid.UUID, title: str) -> None:
        await self.session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id, Conversation.title == DEFAULT_CONVERSATION_TITLE
            )
            .values(title=title[:255], updated_at=func.now())
            .execution_options(synchronize_session=False)
        )

    async def list_messages(
        self, conversation_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[Message], int]:
        conditions = [Message.conversation_id == conversation_id]
        total = (
            await self.session.scalar(select(func.count()).select_from(Message).where(*conditions))
            or 0
        )
        stmt = (
            select(Message)
            .where(*conditions)
            .order_by(Message.created_at.asc(), Message.id)
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.scalars(stmt)).all()), total

    async def recent_messages(self, conversation_id: uuid.UUID, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        items = list((await self.session.scalars(stmt)).all())
        items.reverse()
        return items

    async def get_by_client_message_id(
        self, conversation_id: uuid.UUID, client_message_id: uuid.UUID
    ) -> Message | None:
        stmt = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.client_message_id == client_message_id,
        )
        return await self.session.scalar(stmt)

    async def next_assistant_after(
        self, conversation_id: uuid.UUID, after_id: uuid.UUID, after_created
    ) -> Message | None:
        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.ASSISTANT.value,
                Message.created_at >= after_created,
                Message.id != after_id,
            )
            .order_by(Message.created_at.asc(), Message.id)
            .limit(1)
        )
        return await self.session.scalar(stmt)

    def add_message(
        self,
        *,
        conversation_id: uuid.UUID,
        owner_id: uuid.UUID,
        role: MessageRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
        client_message_id: uuid.UUID | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            owner_id=owner_id,
            role=role.value,
            content=content,
            citations=citations,
            model_name=model_name,
            client_message_id=client_message_id,
        )
        self.session.add(message)
        return message


DEFAULT_CONVERSATION_TITLE = "New conversation"


def title_from_question(content: str) -> str:
    collapsed = " ".join(content.split())
    if len(collapsed) <= 80:
        return collapsed or DEFAULT_CONVERSATION_TITLE
    return collapsed[:77].rstrip() + "..."
