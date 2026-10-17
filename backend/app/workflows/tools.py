"""Workflow retrieval tools (each call uses its own DB session)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import EmbeddingError, RetrievalError
from app.db.models import MessageRole
from app.repositories.conversation_repository import ConversationRepository
from app.services.evidence_merger import chunks_to_evidence, messages_to_evidence
from app.services.retrieval_service import RetrievalContext, RetrievalService
from app.schemas.playground import PlaygroundRetrievalScope
from app.workflows.schemas import CompactMessage, EvidenceItem, QuerySource, RetrievalStrategy

logger = logging.getLogger(__name__)


def _validate_focus_chunk_ids(
    allowed: Sequence[uuid.UUID], focus: Sequence[uuid.UUID] | None
) -> list[uuid.UUID]:
    if not focus:
        return []
    allowed_set = set(allowed)
    return [cid for cid in focus if cid in allowed_set]


class DocumentSearchTool:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        retrieval: RetrievalService,
    ):
        self._session_factory = session_factory
        self._retrieval = retrieval

    async def search(
        self,
        owner_id: uuid.UUID,
        query: str,
        *,
        strategy: RetrievalStrategy,
        retrieval_attempt: int,
        top_k: int | None = None,
        max_candidates: int | None = None,
        focus_chunk_ids: Sequence[uuid.UUID] | None = None,
        prior_chunk_ids: Sequence[uuid.UUID] | None = None,
        retrieval_context: RetrievalContext | None = None,
    ) -> list[EvidenceItem]:
        focus = _validate_focus_chunk_ids(list(prior_chunk_ids or []), focus_chunk_ids)
        ctx = retrieval_context or RetrievalContext.twin()
        try:
            async with self._session_factory() as session:
                chunks = await self._retrieval.retrieve(
                    session,
                    owner_id,
                    query,
                    strategy=strategy,
                    top_k=top_k,
                    max_candidates=max_candidates,
                    focus_chunk_ids=focus or None,
                    retrieval_context=ctx,
                )
        except EmbeddingError as exc:
            raise RetrievalError(str(exc.message)) from exc
        except Exception as exc:
            logger.warning("document_search_failed", extra={"exc_type": type(exc).__name__})
            raise RetrievalError("Document retrieval failed.") from exc
        return chunks_to_evidence(chunks, retrieval_attempt=retrieval_attempt)


class ConversationHistoryTool:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def search(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        messages: list[CompactMessage],
        *,
        query: str,
        limit: int = 6,
    ) -> list[EvidenceItem]:
        del query  # keyword filter optional; use bounded recent messages
        try:
            async with self._session_factory() as session:
                repo = ConversationRepository(session)
                conv = await repo.get_for_owner(conversation_id, owner_id)
                if conv is None:
                    return []
        except Exception as exc:
            logger.warning("conversation_search_failed", extra={"exc_type": type(exc).__name__})
            raise RetrievalError("Conversation history retrieval failed.") from exc
        filtered = [
            m
            for m in messages
            if m.role in (MessageRole.USER.value, MessageRole.ASSISTANT.value)
        ]
        return messages_to_evidence(filtered, conversation_id=conversation_id, max_messages=limit)


async def parallel_retrieve(
    *,
    sources: list[str],
    document_tool: DocumentSearchTool,
    conversation_tool: ConversationHistoryTool,
    owner_id: uuid.UUID,
    conversation_id: uuid.UUID,
    query: str,
    strategy: RetrievalStrategy,
    retrieval_attempt: int,
    recent_messages: list[CompactMessage],
    focus_chunk_ids: Sequence[uuid.UUID] | None,
    prior_document_evidence: list[EvidenceItem],
    retrieval_context: RetrievalContext | None = None,
) -> tuple[list[EvidenceItem], list[EvidenceItem], str | None, str | None]:
    doc_error: str | None = None
    conv_error: str | None = None
    prior_ids = [item.chunk_id for item in prior_document_evidence if item.chunk_id]

    async def run_docs() -> list[EvidenceItem]:
        nonlocal doc_error
        if QuerySource.DOCUMENTS.value not in sources:
            return []
        try:
            return await document_tool.search(
                owner_id,
                query,
                strategy=strategy,
                retrieval_attempt=retrieval_attempt,
                focus_chunk_ids=focus_chunk_ids,
                prior_chunk_ids=prior_ids,
                retrieval_context=retrieval_context,
            )
        except RetrievalError as exc:
            doc_error = exc.message
            return []

    async def run_conv() -> list[EvidenceItem]:
        nonlocal conv_error
        if QuerySource.CONVERSATION_HISTORY.value not in sources:
            return []
        try:
            return await conversation_tool.search(
                owner_id,
                conversation_id,
                recent_messages,
                query=query,
            )
        except RetrievalError as exc:
            conv_error = exc.message
            return []

    doc_items, conv_items = await asyncio.gather(run_docs(), run_conv())
    return doc_items, conv_items, doc_error, conv_error
