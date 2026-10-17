"""Persistent owner memory CRUD and semantic search."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models.memory import MemoryStatus, OwnerMemory
from app.repositories.memory_repository import MemoryRepository, normalize_memory_content
from app.schemas.memory import MemoryRead, ProposedMemory
from app.services.embedding_service import Embedder

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(sk-[a-zA-Z0-9]{10,}|api[_-]?key\s*[:=])", re.I)


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    id: uuid.UUID
    memory_type: str
    content: str
    source_type: str
    source_conversation_id: uuid.UUID | None


class MemoryService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        *,
        embedding_model: str,
        top_k: int,
        min_importance: float,
        max_records: int,
        enabled: bool,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._min_importance = min_importance
        self._max_records = max_records
        self._enabled = enabled

    async def search(self, owner_id: uuid.UUID, query: str) -> list[RetrievedMemory]:
        if not self._enabled or not query.strip():
            return []
        vectors = await self._embedder.embed([query])
        async with self._session_factory() as session:
            repo = MemoryRepository(session)
            rows = await repo.search_semantic(
                owner_id,
                vectors[0],
                self._embedding_model,
                top_k=self._top_k,
                min_importance=self._min_importance,
            )
            await repo.touch_accessed([r.id for r in rows])
            await session.commit()
        return [
            RetrievedMemory(
                id=r.id,
                memory_type=r.memory_type,
                content=r.content,
                source_type=r.source_type,
                source_conversation_id=r.source_conversation_id,
            )
            for r in rows
        ]

    async def persist_proposed(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        proposed: ProposedMemory,
    ) -> OwnerMemory | None:
        if _SECRET_RE.search(proposed.content):
            return None
        if proposed.source_type == "assistant_inference" and proposed.confidence < 0.85:
            return None
        normalized = normalize_memory_content(proposed.content)
        async with self._session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            existing = await repo.find_by_normalized(owner_id, normalized)
            if existing is not None:
                return existing
            _, total = await repo.list_for_owner(owner_id, limit=1, offset=0)
            if total >= self._max_records:
                logger.info("memory_cap_reached", extra={"owner_id": str(owner_id)})
                return None
            vectors = await self._embedder.embed([proposed.content])
            memory = OwnerMemory(
                owner_id=owner_id,
                memory_type=proposed.memory_type,
                content=proposed.content.strip(),
                normalized_content=normalized,
                source_type=proposed.source_type,
                source_conversation_id=conversation_id,
                source_message_ids=list(proposed.source_message_ids),
                confidence=proposed.confidence,
                importance=proposed.importance,
                embedding=vectors[0],
                embedding_model=self._embedding_model,
                supersedes_memory_id=proposed.supersedes_memory_id,
                expires_at=proposed.expires_at,
            )
            if proposed.supersedes_memory_id:
                await repo.set_status(
                    proposed.supersedes_memory_id, owner_id, MemoryStatus.SUPERSEDED.value
                )
            repo.add(memory)
            await session.flush()
            return memory

    async def list_memories(
        self, owner_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[MemoryRead], int]:
        async with self._session_factory() as session:
            repo = MemoryRepository(session)
            rows, total = await repo.list_for_owner(owner_id, limit=limit, offset=offset)
        return [MemoryRead.model_validate(r) for r in rows], total

    async def deactivate(self, owner_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
        async with self._session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            return await repo.set_status(memory_id, owner_id, MemoryStatus.DELETED.value)
