"""Persistence for documents and chunks.

The repository never commits; callers own transaction boundaries. Every
user-facing lookup is scoped by ``owner_id``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import ColumnElement, delete, func, insert, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk, DocumentStatus, DocumentVisibility
from app.schemas.playground import PlaygroundRetrievalScope
from app.services.chunking_service import TextChunk

MAX_ERROR_MESSAGE_LENGTH = 500


def ready_filter() -> ColumnElement[bool]:
    return Document.status == DocumentStatus.READY.value


def playground_retrieval_filter(
    scope: PlaygroundRetrievalScope,
    *,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> ColumnElement[bool]:
    """SQL predicate for Playground vector/fulltext retrieval (caller adds owner_id)."""
    ready = ready_filter()
    if scope == PlaygroundRetrievalScope.GLOBAL_ONLY:
        return twin_retrievable_filter()
    if scope == PlaygroundRetrievalScope.PRIVATE_ONLY:
        return (Document.visibility == DocumentVisibility.PRIVATE.value) & ready
    if scope == PlaygroundRetrievalScope.COMBINED:
        return (
            Document.visibility.in_(
                [DocumentVisibility.GLOBAL.value, DocumentVisibility.PRIVATE.value]
            )
            & ready
        )
    if scope == PlaygroundRetrievalScope.SELECTED:
        if not document_ids:
            raise ValueError("document_ids required for selected scope")
        return ready & Document.id.in_(list(document_ids))
    raise ValueError(f"Unknown playground scope: {scope}")


def twin_retrievable_filter() -> ColumnElement[bool]:
    """Documents the AI Twin may read: global and fully ingested only.

    Private documents are reserved for the future Playground and must never be
    reachable through AI Twin retrieval. Use this in every Twin-facing query.
    """
    return (Document.visibility == DocumentVisibility.GLOBAL.value) & (
        Document.status == DocumentStatus.READY.value
    )


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -- reads ---------------------------------------------------------------

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def get_for_owner(self, document_id: uuid.UUID, owner_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
        return await self.session.scalar(stmt)

    async def find_by_hash(self, owner_id: uuid.UUID, file_hash: str) -> Document | None:
        stmt = select(Document).where(
            Document.owner_id == owner_id, Document.file_hash == file_hash
        )
        return await self.session.scalar(stmt)

    async def list_for_owner(
        self,
        owner_id: uuid.UUID,
        *,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
        visibility: DocumentVisibility | None = None,
        search: str | None = None,
    ) -> tuple[list[Document], int]:
        conditions = [Document.owner_id == owner_id]
        if status is not None:
            conditions.append(Document.status == status.value)
        if visibility is not None:
            conditions.append(Document.visibility == visibility.value)
        if search:
            pattern = f"%{search.strip()}%"
            conditions.append(
                or_(Document.title.ilike(pattern), Document.original_filename.ilike(pattern))
            )
        total = (
            await self.session.scalar(select(func.count()).select_from(Document).where(*conditions))
            or 0
        )
        stmt = (
            select(Document)
            .where(*conditions)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
        items = list((await self.session.scalars(stmt)).all())
        return items, total

    async def count_chunks(self, document_id: uuid.UUID) -> int:
        stmt = select(func.count()).where(DocumentChunk.document_id == document_id)
        return await self.session.scalar(stmt) or 0

    # -- writes --------------------------------------------------------------

    def add(self, document: Document) -> None:
        self.session.add(document)

    async def claim_for_processing(
        self, document_id: uuid.UUID, stale_before: datetime
    ) -> Document | None:
        """Atomically move a document into ``processing``.

        Succeeds only for ``uploaded``/``failed`` documents, or ``processing`` ones whose
        claim is older than ``stale_before`` (a crashed worker). Concurrent callers race
        on the row lock; exactly one gets the row back.
        """
        stmt = (
            update(Document)
            .where(
                Document.id == document_id,
                or_(
                    Document.status.in_(
                        [DocumentStatus.UPLOADED.value, DocumentStatus.FAILED.value]
                    ),
                    (Document.status == DocumentStatus.PROCESSING.value)
                    & (Document.processing_started_at < stale_before),
                ),
            )
            .values(
                status=DocumentStatus.PROCESSING.value,
                processing_started_at=func.now(),
                ingestion_attempts=Document.ingestion_attempts + 1,
                error_message=None,
                updated_at=func.now(),
            )
            .returning(Document)
            .execution_options(synchronize_session=False)
        )
        return await self.session.scalar(stmt)

    async def complete_ingestion(
        self,
        document_id: uuid.UUID,
        *,
        claimed_at: datetime,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        page_count: int,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> bool:
        """Replace all chunks and mark the document ready, in the caller's transaction.

        ``claimed_at`` is the ``processing_started_at`` returned by the claim and acts as a
        fencing token. Returns False (and writes nothing) if the document vanished, is no
        longer ``processing`` (e.g. being deleted), or was reclaimed by a newer run.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        locked = await self.session.execute(
            select(Document.status, Document.processing_started_at)
            .where(Document.id == document_id)
            .with_for_update()
        )
        row = locked.one_or_none()
        if row is None or row.status != DocumentStatus.PROCESSING.value:
            return False
        if row.processing_started_at != claimed_at:
            return False

        # Remove leftovers from any previous (partial or complete) run first, so a retry
        # can never produce duplicate chunk rows.
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        if chunks:
            await self.session.execute(
                insert(DocumentChunk),
                [
                    {
                        "id": uuid.uuid4(),
                        "document_id": document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "token_count": chunk.token_count,
                        "embedding": list(vector),
                    }
                    for chunk, vector in zip(chunks, embeddings, strict=True)
                ],
            )
        await self.session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=DocumentStatus.READY.value,
                page_count=page_count,
                chunk_count=len(chunks),
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
                error_message=None,
                processed_at=func.now(),
                processing_started_at=None,
                updated_at=func.now(),
            )
            .execution_options(synchronize_session=False)
        )
        return True

    async def mark_failed(
        self,
        document_id: uuid.UUID,
        message: str,
        *,
        claimed_at: datetime,
        page_count: int | None = None,
    ) -> bool:
        values: dict[str, object] = {
            "status": DocumentStatus.FAILED.value,
            "error_message": message[:MAX_ERROR_MESSAGE_LENGTH],
            "processing_started_at": None,
            "updated_at": func.now(),
        }
        if page_count is not None:
            values["page_count"] = page_count
        result = await self.session.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.status == DocumentStatus.PROCESSING.value,
                Document.processing_started_at == claimed_at,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount > 0

    async def reset_for_retry(
        self, document_id: uuid.UUID, owner_id: uuid.UUID, stale_before: datetime
    ) -> bool:
        result = await self.session.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.owner_id == owner_id,
                or_(
                    # 'uploaded' covers jobs lost by a restart; duplicate enqueues are harmless
                    # because only one run can win the claim.
                    Document.status.in_(
                        [DocumentStatus.FAILED.value, DocumentStatus.UPLOADED.value]
                    ),
                    (Document.status == DocumentStatus.PROCESSING.value)
                    & (Document.processing_started_at < stale_before),
                ),
            )
            .values(
                status=DocumentStatus.UPLOADED.value,
                error_message=None,
                processing_started_at=None,
                updated_at=func.now(),
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount > 0

    async def mark_deleting(self, document_id: uuid.UUID, owner_id: uuid.UUID) -> Document | None:
        stmt = (
            update(Document)
            .where(Document.id == document_id, Document.owner_id == owner_id)
            .values(status=DocumentStatus.DELETING.value, updated_at=func.now())
            .returning(Document)
            .execution_options(synchronize_session=False)
        )
        return await self.session.scalar(stmt)

    async def delete(self, document_id: uuid.UUID) -> None:
        # document_chunks rows are removed by ON DELETE CASCADE.
        await self.session.execute(delete(Document).where(Document.id == document_id))

    async def update_visibility(
        self, document_id: uuid.UUID, owner_id: uuid.UUID, visibility: DocumentVisibility
    ) -> Document | None:
        stmt = (
            update(Document)
            .where(Document.id == document_id, Document.owner_id == owner_id)
            .values(visibility=visibility.value, updated_at=func.now())
            .returning(Document)
            .execution_options(synchronize_session=False)
        )
        return await self.session.scalar(stmt)
