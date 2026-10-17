"""Owner-scoped persistent memory persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory import MemoryStatus, OwnerMemory


def normalize_memory_content(text: str) -> str:
    return " ".join(text.lower().split())[:4000]


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_for_owner(self, memory_id: uuid.UUID, owner_id: uuid.UUID) -> OwnerMemory | None:
        stmt = select(OwnerMemory).where(
            OwnerMemory.id == memory_id, OwnerMemory.owner_id == owner_id
        )
        return await self.session.scalar(stmt)

    async def list_for_owner(
        self, owner_id: uuid.UUID, *, limit: int, offset: int, active_only: bool = True
    ) -> tuple[list[OwnerMemory], int]:
        conditions = [OwnerMemory.owner_id == owner_id]
        if active_only:
            conditions.append(OwnerMemory.status == MemoryStatus.ACTIVE.value)
        total = (
            await self.session.scalar(
                select(func.count()).select_from(OwnerMemory).where(*conditions)
            )
            or 0
        )
        stmt = (
            select(OwnerMemory)
            .where(*conditions)
            .order_by(OwnerMemory.importance.desc(), OwnerMemory.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.scalars(stmt)).all()), total

    def add(self, memory: OwnerMemory) -> None:
        self.session.add(memory)

    async def search_semantic(
        self,
        owner_id: uuid.UUID,
        query_vec: list[float],
        embedding_model: str,
        *,
        top_k: int,
        min_importance: float,
    ) -> list[OwnerMemory]:
        now = datetime.now(UTC)
        distance = OwnerMemory.embedding.cosine_distance(query_vec)
        stmt = (
            select(OwnerMemory)
            .where(
                OwnerMemory.owner_id == owner_id,
                OwnerMemory.status == MemoryStatus.ACTIVE.value,
                OwnerMemory.embedding_model == embedding_model,
                OwnerMemory.importance >= min_importance,
                (OwnerMemory.expires_at.is_(None)) | (OwnerMemory.expires_at > now),
            )
            .order_by(distance)
            .limit(top_k)
        )
        return list((await self.session.scalars(stmt)).all())

    async def find_by_normalized(
        self, owner_id: uuid.UUID, normalized: str
    ) -> OwnerMemory | None:
        stmt = select(OwnerMemory).where(
            OwnerMemory.owner_id == owner_id,
            OwnerMemory.normalized_content == normalized,
            OwnerMemory.status == MemoryStatus.ACTIVE.value,
        )
        return await self.session.scalar(stmt)

    async def set_status(
        self, memory_id: uuid.UUID, owner_id: uuid.UUID, status: str
    ) -> bool:
        result = await self.session.execute(
            update(OwnerMemory)
            .where(OwnerMemory.id == memory_id, OwnerMemory.owner_id == owner_id)
            .values(status=status, updated_at=func.now())
        )
        return result.rowcount > 0

    async def touch_accessed(self, memory_ids: list[uuid.UUID]) -> None:
        if not memory_ids:
            return
        await self.session.execute(
            update(OwnerMemory)
            .where(OwnerMemory.id.in_(memory_ids))
            .values(last_accessed_at=func.now())
        )
