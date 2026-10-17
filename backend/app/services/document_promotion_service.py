from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError, ValidationFailedError
from app.db.models import DocumentStatus, DocumentVisibility
from app.db.models.playground import DocumentPromotionAudit
from app.repositories.document_repository import DocumentRepository
from app.repositories.playground_repository import DocumentPromotionAuditRepository


class DocumentPromotionService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def promote_to_global(
        self,
        owner_id: uuid.UUID,
        document_id: uuid.UUID,
        *,
        request_id: str | None = None,
    ):
        async with self._session_factory() as session, session.begin():
            repo = DocumentRepository(session)
            document = await repo.get_for_owner(document_id, owner_id)
            if document is None:
                raise NotFoundError("Document not found.")
            previous = document.visibility
            if previous == DocumentVisibility.GLOBAL.value:
                DocumentPromotionAuditRepository(session).add(
                    DocumentPromotionAudit(
                        id=uuid.uuid4(),
                        document_id=document_id,
                        actor_id=owner_id,
                        previous_visibility=previous,
                        new_visibility=DocumentVisibility.GLOBAL.value,
                        outcome="already_global",
                        request_id=request_id,
                    )
                )
                return document
            if document.status != DocumentStatus.READY.value:
                raise ValidationFailedError(
                    "Only ready documents can be promoted to global scope.",
                    code="document_not_ready",
                )
            updated = await repo.update_visibility(
                document_id, owner_id, DocumentVisibility.GLOBAL
            )
            if updated is None:
                raise NotFoundError("Document not found.")
            DocumentPromotionAuditRepository(session).add(
                DocumentPromotionAudit(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    actor_id=owner_id,
                    previous_visibility=previous,
                    new_visibility=DocumentVisibility.GLOBAL.value,
                    outcome="success",
                    request_id=request_id,
                )
            )
            return updated
