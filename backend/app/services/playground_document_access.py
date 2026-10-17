"""Validate Playground document scope before retrieval."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationFailedError
from app.db.models import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.schemas.playground import PlaygroundRetrievalScope


class PlaygroundDocumentAccessService:
    async def validate_scope(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        scope: PlaygroundRetrievalScope,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> list[uuid.UUID]:
        if scope != PlaygroundRetrievalScope.SELECTED:
            return []
        if not document_ids:
            raise ValidationFailedError(
                "document_ids is required when scope is 'selected'.",
                code="missing_document_ids",
            )
        repo = DocumentRepository(session)
        validated: list[uuid.UUID] = []
        for doc_id in document_ids:
            doc = await repo.get_for_owner(doc_id, owner_id)
            if doc is None:
                raise NotFoundError("Document not found.")
            if doc.status != DocumentStatus.READY.value:
                raise ValidationFailedError(
                    f"Document '{doc_id}' is not ready for retrieval.",
                    code="document_not_ready",
                )
            validated.append(doc_id)
        return validated
