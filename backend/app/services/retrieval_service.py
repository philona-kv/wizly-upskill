"""Authorized pgvector retrieval for the AI Twin.

SQL already restricts results to the caller's ready **global** documents. Private
chunks are never loaded into Python for filtering.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

RRF_RANK_CONSTANT = 60

from sqlalchemy import func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.repositories.document_repository import (
    playground_retrieval_filter,
    twin_retrievable_filter,
)
from app.schemas.playground import PlaygroundRetrievalScope
from app.services.embedding_service import Embedder
from app.workflows.schemas import RetrievalStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    mode: Literal["twin", "playground"] = "twin"
    scope: PlaygroundRetrievalScope = PlaygroundRetrievalScope.GLOBAL_ONLY
    document_ids: tuple[uuid.UUID, ...] = ()

    @classmethod
    def twin(cls) -> RetrievalContext:
        return cls(mode="twin")

    @classmethod
    def playground(
        cls,
        scope: PlaygroundRetrievalScope,
        *,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> RetrievalContext:
        ids = tuple(document_ids or ())
        return cls(mode="playground", scope=scope, document_ids=ids)


def retrieval_filters(
    owner_id: uuid.UUID,
    embedding_model: str,
    *,
    context: RetrievalContext | None = None,
):
    """SQL predicates for retrieval. Default context is Twin (global + ready only)."""
    ctx = context or RetrievalContext.twin()
    if ctx.mode == "twin":
        visibility = twin_retrievable_filter()
    else:
        visibility = playground_retrieval_filter(
            ctx.scope, document_ids=list(ctx.document_ids) or None
        )
    return (
        Document.owner_id == owner_id,
        visibility,
        Document.embedding_model == embedding_model,
    )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    original_filename: str
    content: str
    page_start: int
    page_end: int
    chunk_index: int
    distance: float


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    top_k: int,
    rrf_k: int = RRF_RANK_CONSTANT,
) -> list[RetrievedChunk]:
    """Merge ranked lists with reciprocal rank fusion (chunk may appear in multiple lists)."""
    scores: dict[uuid.UUID, float] = {}
    by_id: dict[uuid.UUID, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            by_id[chunk.chunk_id] = chunk
    if not scores:
        return []
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id, score in ordered[:top_k]:
        base = by_id[chunk_id]
        fused.append(replace(base, distance=1.0 / score))
    return fused


def _rows_to_chunks(rows) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            document_title=row.title,
            original_filename=row.original_filename,
            content=row.content,
            page_start=row.page_start,
            page_end=row.page_end,
            chunk_index=row.chunk_index,
            distance=float(row.distance) if row.distance is not None else 0.0,
        )
        for row in rows
    ]


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        *,
        top_k: int,
        max_candidates: int,
        embedding_model: str,
    ) -> None:
        self._embedder = embedder
        self._top_k = top_k
        self._max_candidates = max_candidates
        self._embedding_model = embedding_model

    async def retrieve(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        query: str,
        *,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        top_k: int | None = None,
        max_candidates: int | None = None,
        focus_chunk_ids: Sequence[uuid.UUID] | None = None,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[RetrievedChunk]:
        ctx = retrieval_context or RetrievalContext.twin()
        top_k = top_k or self._top_k
        max_candidates = max_candidates or self._max_candidates
        if strategy == RetrievalStrategy.FULLTEXT:
            return await self.retrieve_fulltext(
                session,
                owner_id,
                query,
                limit=max(top_k, max_candidates),
                retrieval_context=ctx,
            )
        if strategy == RetrievalStrategy.HYBRID:
            return await self.retrieve_hybrid(
                session,
                owner_id,
                query,
                top_k=top_k,
                max_candidates=max_candidates,
                retrieval_context=ctx,
            )
        if strategy == RetrievalStrategy.NEIGHBOR_EXPAND and focus_chunk_ids:
            return await self.retrieve_neighbors(
                session, owner_id, list(focus_chunk_ids), retrieval_context=ctx
            )
        if strategy == RetrievalStrategy.SECTION_FOCUS and focus_chunk_ids:
            return await self.retrieve_neighbors(
                session,
                owner_id,
                list(focus_chunk_ids),
                window=2,
                retrieval_context=ctx,
            )
        if strategy == RetrievalStrategy.BROADEN_SEMANTIC:
            top_k = min(top_k + 4, 50)
            max_candidates = min(int(max_candidates * 1.5), 200)
        return await self._semantic(
            session, owner_id, query, top_k, max_candidates, retrieval_context=ctx
        )

    async def _semantic_candidates(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        query: str,
        limit: int,
        *,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[RetrievedChunk]:
        vectors = await self._embedder.embed([query])
        query_vec = vectors[0]
        distance = DocumentChunk.embedding.cosine_distance(query_vec)
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title,
                Document.original_filename,
                DocumentChunk.content,
                DocumentChunk.page_start,
                DocumentChunk.page_end,
                DocumentChunk.chunk_index,
                distance.label("distance"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                *retrieval_filters(
                    owner_id, self._embedding_model, context=retrieval_context
                )
            )
            .order_by(distance)
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        return _rows_to_chunks(rows)

    async def _semantic(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        query: str,
        top_k: int,
        max_candidates: int,
        *,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[RetrievedChunk]:
        hits = await self._semantic_candidates(
            session,
            owner_id,
            query,
            max_candidates,
            retrieval_context=retrieval_context,
        )
        selected = hits[:top_k]
        logger.info(
            "retrieval_complete",
            extra={"candidates": len(hits), "selected": len(selected), "owner_id": str(owner_id)},
        )
        return selected

    async def retrieve_hybrid(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        query: str,
        *,
        top_k: int,
        max_candidates: int,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[RetrievedChunk]:
        ctx = retrieval_context or RetrievalContext.twin()
        limit = max(top_k, max_candidates)
        semantic_hits, fulltext_hits = await asyncio.gather(
            self._semantic_candidates(
                session, owner_id, query, limit, retrieval_context=ctx
            ),
            self.retrieve_fulltext(
                session, owner_id, query, limit=limit, retrieval_context=ctx
            ),
        )
        fused = reciprocal_rank_fusion([semantic_hits, fulltext_hits], top_k=top_k)
        logger.info(
            "hybrid_retrieval_complete",
            extra={
                "semantic": len(semantic_hits),
                "fulltext": len(fulltext_hits),
                "selected": len(fused),
                "owner_id": str(owner_id),
            },
        )
        return fused

    async def retrieve_fulltext(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        query: str,
        *,
        limit: int = 20,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[RetrievedChunk]:
        ts_query = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank(
            func.to_tsvector("english", DocumentChunk.content),
            ts_query,
        ).label("distance")
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title,
                Document.original_filename,
                DocumentChunk.content,
                DocumentChunk.page_start,
                DocumentChunk.page_end,
                DocumentChunk.chunk_index,
                rank,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                *retrieval_filters(
                    owner_id, self._embedding_model, context=retrieval_context
                ),
                func.to_tsvector("english", DocumentChunk.content).op("@@")(ts_query),
            )
            .order_by(rank.desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        return _rows_to_chunks(rows)

    async def retrieve_neighbors(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
        *,
        window: int = 1,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        anchor_stmt = select(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
        ).where(DocumentChunk.id.in_(chunk_ids))
        anchors = (await session.execute(anchor_stmt)).all()
        if not anchors:
            return []
        index_predicates = []
        doc_ids: set[uuid.UUID] = set()
        for row in anchors:
            doc_ids.add(row.document_id)
            for offset in range(-window, window + 1):
                index_predicates.append(
                    (DocumentChunk.document_id == row.document_id)
                    & (DocumentChunk.chunk_index == row.chunk_index + offset)
                )
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.title,
                Document.original_filename,
                DocumentChunk.content,
                DocumentChunk.page_start,
                DocumentChunk.page_end,
                DocumentChunk.chunk_index,
                literal(0.0).label("distance"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                *retrieval_filters(
                    owner_id, self._embedding_model, context=retrieval_context
                ),
                DocumentChunk.document_id.in_(doc_ids),
                or_(*index_predicates),
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        rows = (await session.execute(stmt)).all()
        return _rows_to_chunks(rows)
