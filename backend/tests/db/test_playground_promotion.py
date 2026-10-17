from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.db.models import DocumentVisibility
from app.db.models.playground import DocumentPromotionAudit
from app.services.document_promotion_service import DocumentPromotionService
from app.services.retrieval_service import RetrievalService
from tests.db.conftest import OWNER_A, OWNER_B
from tests.db.test_conversations import UNIT

pytestmark = pytest.mark.db


async def test_promote_private_to_global(session_factory, embedder) -> None:
    doc_id = uuid.uuid4()
    async with session_factory() as session, session.begin():
        from app.db.models import Document, DocumentChunk
        from app.db.models.document_chunk import EMBEDDING_DIMENSIONS

        session.add(
            Document(
                id=doc_id,
                owner_id=OWNER_A,
                title="Private",
                original_filename="p.pdf",
                storage_path=f"{OWNER_A}/{doc_id}.pdf",
                mime_type="application/pdf",
                file_size_bytes=10,
                file_hash=uuid.uuid4().hex,
                visibility="private",
                status="ready",
                chunk_count=1,
                embedding_model="text-embedding-3-small",
                embedding_dimensions=EMBEDDING_DIMENSIONS,
            )
        )
        session.add(
            DocumentChunk(
                document_id=doc_id,
                chunk_index=0,
                content="promoted secret policy",
                page_start=1,
                page_end=1,
                token_count=5,
                embedding=UNIT,
            )
        )

    service = DocumentPromotionService(session_factory)
    updated = await service.promote_to_global(OWNER_A, doc_id)
    assert updated.visibility == DocumentVisibility.GLOBAL.value

    retrieval = RetrievalService(
        embedder, top_k=5, max_candidates=10, embedding_model="text-embedding-3-small"
    )
    async with session_factory() as session:
        twin_hits = await retrieval.retrieve(session, OWNER_A, "promoted secret")
        assert len(twin_hits) >= 1

    async with session_factory() as session:
        audits = list(
            (await session.scalars(select(DocumentPromotionAudit))).all()
        )
    assert any(a.outcome == "success" for a in audits)


async def test_cannot_promote_other_owner(client, session_factory) -> None:
    doc_id = uuid.uuid4()
    async with session_factory() as session, session.begin():
        from app.db.models import Document

        session.add(
            Document(
                id=doc_id,
                owner_id=OWNER_B,
                title="B private",
                original_filename="b.pdf",
                storage_path=f"{OWNER_B}/{doc_id}.pdf",
                mime_type="application/pdf",
                file_size_bytes=10,
                file_hash=uuid.uuid4().hex,
                visibility="private",
                status="ready",
                chunk_count=0,
            )
        )
    response = await client.post(f"/api/v1/playground/documents/{doc_id}/promote")
    assert response.status_code == 404
