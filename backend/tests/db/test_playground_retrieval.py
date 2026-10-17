from __future__ import annotations

import uuid

import pytest

from app.schemas.playground import PlaygroundRetrievalScope
from app.services.retrieval_service import RetrievalContext, RetrievalService
from tests.db.conftest import OWNER_A, OWNER_B
from tests.db.test_conversations import UNIT, _ready_document

pytestmark = pytest.mark.db


@pytest.fixture
def retrieval(embedder) -> RetrievalService:
    return RetrievalService(
        embedder,
        top_k=5,
        max_candidates=20,
        embedding_model="text-embedding-3-small",
    )


async def test_owner_retrieves_private_in_playground(session_factory, retrieval) -> None:
    await _ready_document(
        session_factory,
        visibility="private",
        content="SECRET private playground text 48 hours.",
    )
    async with session_factory() as session:
        hits = await retrieval.retrieve(
            session,
            OWNER_A,
            "private playground",
            retrieval_context=RetrievalContext.playground(
                PlaygroundRetrievalScope.PRIVATE_ONLY
            ),
        )
    assert len(hits) == 1
    assert "SECRET" in hits[0].content


async def test_other_owner_cannot_retrieve_private(session_factory, retrieval) -> None:
    await _ready_document(
        session_factory,
        owner_id=OWNER_B,
        visibility="private",
        content="OWNER B secret content",
        filename="b.pdf",
    )
    async with session_factory() as session:
        hits = await retrieval.retrieve(
            session,
            OWNER_A,
            "OWNER B secret",
            retrieval_context=RetrievalContext.playground(
                PlaygroundRetrievalScope.COMBINED
            ),
        )
    assert hits == []


async def test_global_only_excludes_private(session_factory, retrieval) -> None:
    await _ready_document(session_factory, visibility="global", content="global policy 48 hours")
    await _ready_document(
        session_factory,
        visibility="private",
        content="private only secret",
        filename="priv.pdf",
    )
    async with session_factory() as session:
        hits = await retrieval.retrieve(
            session,
            OWNER_A,
            "policy secret",
            retrieval_context=RetrievalContext.playground(
                PlaygroundRetrievalScope.GLOBAL_ONLY
            ),
        )
    assert len(hits) == 1
    assert "global policy" in hits[0].content


async def test_combined_includes_global_and_private(session_factory, retrieval) -> None:
    await _ready_document(
        session_factory, visibility="global", content="combined global text", title="G"
    )
    await _ready_document(
        session_factory,
        visibility="private",
        content="combined private text",
        filename="p.pdf",
        title="P",
    )
    async with session_factory() as session:
        hits = await retrieval.retrieve(
            session,
            OWNER_A,
            "combined",
            retrieval_context=RetrievalContext.playground(PlaygroundRetrievalScope.COMBINED),
        )
    contents = {h.content for h in hits}
    assert "combined global text" in contents
    assert "combined private text" in contents


async def test_selected_scope_limits_documents(session_factory, retrieval) -> None:
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()
    async with session_factory() as session, session.begin():
        from app.db.models import Document, DocumentChunk
        from app.db.models.document_chunk import EMBEDDING_DIMENSIONS

        for doc_id, text in ((doc_a, "selected A content"), (doc_b, "selected B content")):
            session.add(
                Document(
                    id=doc_id,
                    owner_id=OWNER_A,
                    title="T",
                    original_filename=f"{doc_id}.pdf",
                    storage_path=f"{OWNER_A}/{doc_id}.pdf",
                    mime_type="application/pdf",
                    file_size_bytes=100,
                    file_hash=uuid.uuid4().hex,
                    visibility="global",
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
                    content=text,
                    page_start=1,
                    page_end=1,
                    token_count=10,
                    embedding=UNIT,
                )
            )
    async with session_factory() as session:
        hits = await retrieval.retrieve(
            session,
            OWNER_A,
            "selected",
            retrieval_context=RetrievalContext.playground(
                PlaygroundRetrievalScope.SELECTED, document_ids=[doc_a]
            ),
        )
    assert len(hits) == 1
    assert hits[0].document_id == doc_a


async def test_non_ready_excluded(session_factory, retrieval) -> None:
    await _ready_document(session_factory, status="processing", content="not ready yet")
    async with session_factory() as session:
        hits = await retrieval.retrieve(
            session,
            OWNER_A,
            "not ready",
            retrieval_context=RetrievalContext.playground(
                PlaygroundRetrievalScope.COMBINED
            ),
        )
    assert hits == []
