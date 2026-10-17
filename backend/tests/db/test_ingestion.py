from __future__ import annotations

import asyncio
import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update

from app.db.models import Document, DocumentChunk, DocumentStatus
from app.repositories.document_repository import DocumentRepository, twin_retrievable_filter
from app.services.ingestion_service import GENERIC_FAILURE, IngestionOutcome
from app.services.storage_service import build_storage_path
from tests.conftest import SAMPLE_PARAGRAPH, make_pdf
from tests.db.conftest import OWNER_A

pytestmark = pytest.mark.db

LONG_PDF = make_pdf([SAMPLE_PARAGRAPH * 8, SAMPLE_PARAGRAPH * 8, SAMPLE_PARAGRAPH * 4])


async def _create_document(session_factory, storage, data: bytes = LONG_PDF, **overrides):
    document_id = uuid.uuid4()
    path = build_storage_path(OWNER_A, document_id)
    await storage.upload(path, io.BytesIO(data), len(data), "application/pdf")
    values = {
        "id": document_id,
        "owner_id": OWNER_A,
        "title": "t",
        "original_filename": "t.pdf",
        "storage_path": path,
        "mime_type": "application/pdf",
        "file_size_bytes": len(data),
        "file_hash": uuid.uuid4().hex,
        "visibility": "global",
        "status": DocumentStatus.UPLOADED.value,
        **overrides,
    }
    async with session_factory() as session, session.begin():
        session.add(Document(**values))
    return document_id


async def _load(session_factory, document_id) -> tuple[Document, list[DocumentChunk]]:
    async with session_factory() as session:
        doc = await session.get(Document, document_id)
        chunks = list(
            (
                await session.scalars(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                )
            ).all()
        )
    return doc, chunks


async def test_successful_processing(session_factory, storage, ingestion) -> None:
    document_id = await _create_document(session_factory, storage)
    assert await ingestion.run(document_id) == IngestionOutcome.READY

    doc, chunks = await _load(session_factory, document_id)
    assert doc.status == "ready"
    assert doc.page_count == 3
    assert doc.chunk_count == len(chunks) > 1
    assert doc.embedding_model == "text-embedding-3-small"
    assert doc.embedding_dimensions == 1536
    assert doc.processing_started_at is None
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert all(c.token_count and c.token_count <= 120 for c in chunks)
    assert len(chunks[0].embedding) == 1536


async def test_embedding_failure_marks_failed_without_chunks(
    session_factory, storage, ingestion, embedder
) -> None:
    embedder.fail_times = 1
    document_id = await _create_document(session_factory, storage)
    assert await ingestion.run(document_id) == IngestionOutcome.FAILED
    doc, chunks = await _load(session_factory, document_id)
    assert doc.status == "failed"
    assert "rate limit" in doc.error_message
    assert doc.page_count == 3  # kept for diagnosis
    assert chunks == []


async def test_storage_failure(session_factory, storage, ingestion) -> None:
    document_id = await _create_document(session_factory, storage)
    storage.fail_download = True
    assert await ingestion.run(document_id) == IngestionOutcome.FAILED
    doc, _ = await _load(session_factory, document_id)
    assert doc.status == "failed"
    assert doc.error_message == "Could not reach file storage."


async def test_missing_storage_object(session_factory, storage, ingestion) -> None:
    document_id = await _create_document(session_factory, storage)
    storage.objects.clear()
    assert await ingestion.run(document_id) == IngestionOutcome.FAILED
    doc, _ = await _load(session_factory, document_id)
    assert "missing" in doc.error_message


async def test_database_failure_during_persist(
    session_factory, storage, ingestion, monkeypatch
) -> None:
    document_id = await _create_document(session_factory, storage)

    async def boom(self, *args, **kwargs):
        raise RuntimeError("connection reset: password=hunter2")

    monkeypatch.setattr(DocumentRepository, "complete_ingestion", boom)
    assert await ingestion.run(document_id) == IngestionOutcome.FAILED
    doc, chunks = await _load(session_factory, document_id)
    assert doc.status == "failed"
    assert doc.error_message == GENERIC_FAILURE
    assert "hunter2" not in doc.error_message
    assert chunks == []


async def test_retry_after_failure_succeeds(session_factory, storage, ingestion, embedder) -> None:
    embedder.fail_times = 1
    document_id = await _create_document(session_factory, storage)
    assert await ingestion.run(document_id) == IngestionOutcome.FAILED
    assert await ingestion.run(document_id) == IngestionOutcome.READY
    doc, chunks = await _load(session_factory, document_id)
    assert doc.status == "ready" and doc.error_message is None
    assert doc.ingestion_attempts == 2
    assert doc.chunk_count == len(chunks)


async def test_ready_document_is_not_reprocessed(
    session_factory, storage, ingestion, embedder
) -> None:
    document_id = await _create_document(session_factory, storage)
    await ingestion.run(document_id)
    calls = embedder.calls
    assert await ingestion.run(document_id) == IngestionOutcome.SKIPPED
    assert embedder.calls == calls


async def test_concurrent_runs_process_once(session_factory, storage, ingestion, embedder) -> None:
    document_id = await _create_document(session_factory, storage)
    outcomes = await asyncio.gather(*(ingestion.run(document_id) for _ in range(5)))
    assert sorted(outcomes).count(IngestionOutcome.READY) == 1
    assert outcomes.count(IngestionOutcome.SKIPPED) == 4
    assert embedder.calls == 1
    doc, chunks = await _load(session_factory, document_id)
    assert doc.chunk_count == len(chunks)


async def test_partial_chunks_are_replaced_on_retry(session_factory, storage, ingestion) -> None:
    document_id = await _create_document(session_factory, storage, status="failed")
    async with session_factory() as session, session.begin():
        for i in (0, 1, 7):  # stale leftovers from a hypothetical earlier run
            session.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content="stale",
                    page_start=1,
                    page_end=1,
                    token_count=1,
                    embedding=[0.0] * 1536,
                )
            )
    assert await ingestion.run(document_id) == IngestionOutcome.READY
    doc, chunks = await _load(session_factory, document_id)
    assert all(c.content != "stale" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(doc.chunk_count))


async def test_stale_processing_claim_can_be_reclaimed(session_factory, storage, ingestion) -> None:
    document_id = await _create_document(
        session_factory, storage, status="processing", processing_started_at=datetime.now(UTC)
    )
    assert await ingestion.run(document_id) == IngestionOutcome.SKIPPED
    async with session_factory() as session, session.begin():
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(processing_started_at=datetime.now(UTC) - timedelta(hours=1))
        )
    assert await ingestion.run(document_id) == IngestionOutcome.READY


async def test_deletion_during_processing_is_not_resurrected(
    session_factory, storage, ingestion, embedder
) -> None:
    document_id = await _create_document(session_factory, storage)

    async def mark_deleting() -> None:
        async with session_factory() as session, session.begin():
            await DocumentRepository(session).mark_deleting(document_id, OWNER_A)

    embedder.before_return = mark_deleting
    assert await ingestion.run(document_id) == IngestionOutcome.ABORTED
    doc, chunks = await _load(session_factory, document_id)
    assert doc.status == "deleting"
    assert chunks == []


async def test_superseded_run_cannot_overwrite_newer_claim(
    session_factory, storage, ingestion, embedder
) -> None:
    document_id = await _create_document(session_factory, storage)
    newer_claim = datetime.now(UTC) + timedelta(seconds=5)

    async def reclaimed_by_other_worker() -> None:
        async with session_factory() as session, session.begin():
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(processing_started_at=newer_claim)
            )

    embedder.before_return = reclaimed_by_other_worker
    assert await ingestion.run(document_id) == IngestionOutcome.ABORTED
    doc, chunks = await _load(session_factory, document_id)
    assert doc.status == "processing" and doc.processing_started_at == newer_claim
    assert chunks == []

    # A failure in the superseded run must not flip the newer run's document to failed.
    embedder.before_return = None
    embedder.fail_times = 1
    async with session_factory() as session, session.begin():
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(processing_started_at=datetime.now(UTC) - timedelta(hours=1))
        )
    embedder.before_return = reclaimed_by_other_worker
    assert await ingestion.run(document_id) == IngestionOutcome.ABORTED
    doc, _ = await _load(session_factory, document_id)
    assert doc.status == "processing"


async def test_deleting_document_is_never_claimed(session_factory, storage, ingestion) -> None:
    document_id = await _create_document(session_factory, storage, status="deleting")
    assert await ingestion.run(document_id) == IngestionOutcome.SKIPPED


async def test_twin_filter_excludes_private_and_unready(
    session_factory, storage, ingestion
) -> None:
    public_id = await _create_document(session_factory, storage)
    private_id = await _create_document(
        session_factory, storage, data=make_pdf(["x " * 50]), visibility="private"
    )
    await ingestion.run(public_id)
    await ingestion.run(private_id)
    async with session_factory() as session:
        ids = set(
            (await session.scalars(select(Document.id).where(twin_retrievable_filter()))).all()
        )
        total_chunks = await session.scalar(select(func.count()).select_from(DocumentChunk))
    assert ids == {public_id}
    assert total_chunks > 0
