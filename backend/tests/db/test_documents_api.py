from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from app.db.models import Document, DocumentChunk
from tests.conftest import (
    SAMPLE_PARAGRAPH,
    make_encrypted_pdf,
    make_image_bytes,
    make_image_only_pdf,
    make_pdf,
)
from tests.db.conftest import OWNER_B

pytestmark = pytest.mark.db


async def _upload(
    client,
    data: bytes,
    *,
    name="report.pdf",
    visibility="global",
    title=None,
    content_type="application/pdf",
):
    form = {"visibility": visibility}
    if title is not None:
        form["title"] = title
    return await client.post(
        "/api/v1/documents", files={"file": (name, data, content_type)}, data=form
    )


async def test_health(client) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_checks_database(client) -> None:
    response = await client.get("/api/v1/health/ready")
    body = response.json()
    assert response.status_code == 200, body
    assert body["checks"] == {
        "database": "ok",
        "pgvector": "ok",
        "storage": "ok",
        "embeddings": "ok",
    }


async def test_valid_upload_is_accepted_and_ingested(
    client, sample_pdf_bytes, storage, session_factory
) -> None:
    response = await _upload(client, sample_pdf_bytes, title="  Quarterly   report ")
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["duplicate"] is False
    assert body["status"] == "uploaded"
    document_id = uuid.UUID(body["document_id"])
    assert body["document"]["title"] == "Quarterly report"
    assert "storage_path" not in body["document"]

    [path] = storage.objects
    assert path.endswith(f"/{document_id}.pdf") and "report" not in path

    status = (await client.get(f"/api/v1/documents/{document_id}/status")).json()
    assert status["status"] == "ready"
    assert status["page_count"] == 2
    assert status["chunk_count"] >= 1

    async with session_factory() as session:
        count = await session.scalar(
            select(func.count()).where(DocumentChunk.document_id == document_id)
        )
        doc = await session.get(Document, document_id)
    assert count == status["chunk_count"]
    assert doc.embedding_model == "text-embedding-3-small"
    assert doc.processed_at is not None


async def test_title_defaults_to_filename(client, sample_pdf_bytes) -> None:
    body = (await _upload(client, sample_pdf_bytes, name="../../etc/My Notes.pdf")).json()
    assert body["document"]["title"] == "My Notes"
    assert body["document"]["original_filename"] == "My Notes.pdf"


@pytest.mark.parametrize(
    ("name", "content_type"),
    [("notes.txt", "text/plain"), ("image.gif", "image/gif"), ("doc.pdf", "image/png")],
)
async def test_unsupported_file_type(client, sample_pdf_bytes, name, content_type) -> None:
    response = await _upload(client, sample_pdf_bytes, name=name, content_type=content_type)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


async def test_oversized_file(client, storage) -> None:
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    response = await _upload(client, big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert storage.objects == {}


@pytest.mark.parametrize(
    "data",
    [b"", b"not a pdf at all", b"%PDF-1.7\ngarbage\x00\x01", make_encrypted_pdf()],
    ids=["empty", "no-signature", "corrupted", "encrypted"],
)
async def test_malformed_pdf(client, storage, data) -> None:
    response = await _upload(client, data)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_pdf"
    assert storage.objects == {}


async def test_invalid_visibility(client, sample_pdf_bytes) -> None:
    response = await _upload(client, sample_pdf_bytes, visibility="public")
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("fmt", "name", "content_type", "suffix"),
    [
        ("PNG", "scan.png", "image/png", ".png"),
        ("JPEG", "photo.jpeg", "image/jpeg", ".jpg"),
    ],
)
async def test_valid_image_upload_is_accepted_and_ingested(
    client, storage, session_factory, monkeypatch, fmt, name, content_type, suffix
) -> None:
    monkeypatch.setattr("app.services.image_extractor.ocr_image", lambda _img: SAMPLE_PARAGRAPH * 3)
    data = make_image_bytes(fmt, with_text=True)
    response = await _upload(client, data, name=name, content_type=content_type, title="Scan")
    assert response.status_code == 202, response.text
    body = response.json()
    document_id = uuid.UUID(body["document_id"])
    assert body["document"]["mime_type"] == ("image/png" if fmt == "PNG" else "image/jpeg")
    [path] = storage.objects
    assert path.endswith(f"/{document_id}{suffix}")
    status = (await client.get(f"/api/v1/documents/{document_id}/status")).json()
    assert status["status"] == "ready"
    assert status["page_count"] == 1
    assert status["chunk_count"] >= 1
    async with session_factory() as session:
        doc = await session.get(Document, document_id)
    assert doc.original_filename == name


async def test_malformed_image_is_rejected(client, storage) -> None:
    response = await _upload(
        client, b"\x89PNG\r\n\x1a\nnot-an-image", name="broken.png", content_type="image/png"
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"
    assert storage.objects == {}


async def test_image_without_text_fails_ingestion(client, monkeypatch) -> None:
    monkeypatch.setattr("app.services.image_extractor.ocr_image", lambda _img: "")
    body = (
        await _upload(client, make_image_bytes("PNG"), name="blank.png", content_type="image/png")
    ).json()
    status = (await client.get(f"/api/v1/documents/{body['document_id']}/status")).json()
    assert status["status"] == "failed"
    assert "No extractable text" in status["error_message"]


async def test_scanned_pdf_ingests_when_ocr_returns_text(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.image_extractor.ocr_image",
        lambda _img: "Scanned page text with enough characters for ingestion.",
    )
    body = (await _upload(client, make_image_only_pdf())).json()
    status = (await client.get(f"/api/v1/documents/{body['document_id']}/status")).json()
    assert status["status"] == "ready"


async def test_scanned_pdf_fails_when_ocr_is_empty(client, monkeypatch) -> None:
    monkeypatch.setattr("app.services.image_extractor.ocr_image", lambda _img: "")
    body = (await _upload(client, make_image_only_pdf())).json()
    status = (await client.get(f"/api/v1/documents/{body['document_id']}/status")).json()
    assert status["status"] == "failed"
    assert "No extractable text" in status["error_message"]


async def test_duplicate_upload_returns_existing(
    client, sample_pdf_bytes, storage, job_runner
) -> None:
    first = (await _upload(client, sample_pdf_bytes)).json()
    second = await _upload(client, sample_pdf_bytes, name="copy.pdf")
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first["document_id"]
    assert len(storage.objects) == 1
    assert len(job_runner.enqueued) == 1


async def test_duplicate_with_other_visibility_conflicts(client, sample_pdf_bytes) -> None:
    await _upload(client, sample_pdf_bytes, visibility="global")
    response = await _upload(client, sample_pdf_bytes, visibility="private")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_visibility_mismatch"


async def test_same_file_for_other_owner_is_separate(
    client, sample_pdf_bytes, current_owner
) -> None:
    first = (await _upload(client, sample_pdf_bytes)).json()
    current_owner.owner_id = OWNER_B
    second = await _upload(client, sample_pdf_bytes)
    assert second.status_code == 202
    assert second.json()["document_id"] != first["document_id"]


async def test_storage_failure_on_upload_creates_nothing(
    client, sample_pdf_bytes, storage, session_factory
) -> None:
    storage.fail_upload = True
    response = await _upload(client, sample_pdf_bytes)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "storage_error"
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 0


async def test_listing_with_pagination_and_filters(client, job_runner) -> None:
    job_runner.run_jobs = False
    for i in range(3):
        await _upload(
            client,
            make_pdf([f"{SAMPLE_PARAGRAPH} doc {i}"]),
            name=f"d{i}.pdf",
            visibility="private" if i == 0 else "global",
        )
    page = (await client.get("/api/v1/documents", params={"limit": 2})).json()
    assert page["total"] == 3 and len(page["items"]) == 2
    rest = (await client.get("/api/v1/documents", params={"limit": 2, "offset": 2})).json()
    assert len(rest["items"]) == 1
    private = (await client.get("/api/v1/documents", params={"visibility": "private"})).json()
    assert private["total"] == 1
    uploaded = (await client.get("/api/v1/documents", params={"status": "uploaded"})).json()
    assert uploaded["total"] == 3
    assert "content" not in page["items"][0] and "embedding" not in page["items"][0]


async def test_ownership_enforced_everywhere(client, sample_pdf_bytes, current_owner) -> None:
    document_id = (await _upload(client, sample_pdf_bytes)).json()["document_id"]
    current_owner.owner_id = OWNER_B

    assert (await client.get("/api/v1/documents")).json()["total"] == 0
    for method, url in [
        ("GET", f"/api/v1/documents/{document_id}"),
        ("GET", f"/api/v1/documents/{document_id}/status"),
        ("POST", f"/api/v1/documents/{document_id}/retry"),
        ("DELETE", f"/api/v1/documents/{document_id}"),
    ]:
        response = await client.request(method, url)
        assert response.status_code == 404, (method, url)

    current_owner.owner_id = uuid.UUID("00000000-0000-0000-0000-00000000000a")
    assert (await client.get(f"/api/v1/documents/{document_id}")).status_code == 200


async def test_unknown_document_and_bad_id(client) -> None:
    assert (await client.get(f"/api/v1/documents/{uuid.uuid4()}")).status_code == 404
    assert (await client.get("/api/v1/documents/not-a-uuid")).status_code == 422


async def test_delete_removes_row_chunks_and_object(
    client, sample_pdf_bytes, storage, session_factory
) -> None:
    document_id = (await _upload(client, sample_pdf_bytes)).json()["document_id"]
    response = await client.delete(f"/api/v1/documents/{document_id}")
    assert response.status_code == 204
    assert storage.objects == {}
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(DocumentChunk)) == 0
        assert await session.scalar(select(func.count()).select_from(Document)) == 0
    # Repeating the delete is safe.
    assert (await client.delete(f"/api/v1/documents/{document_id}")).status_code == 404


async def test_delete_with_storage_failure_can_be_resumed(
    client, sample_pdf_bytes, storage, session_factory
) -> None:
    document_id = (await _upload(client, sample_pdf_bytes)).json()["document_id"]
    storage.fail_delete = True
    response = await client.delete(f"/api/v1/documents/{document_id}")
    assert response.status_code == 502
    status = (await client.get(f"/api/v1/documents/{document_id}/status")).json()
    assert status["status"] == "deleting"

    # Re-uploading the same file while deletion is pending is refused.
    assert (await _upload(client, sample_pdf_bytes)).status_code == 409

    storage.fail_delete = False
    assert (await client.delete(f"/api/v1/documents/{document_id}")).status_code == 204
    assert storage.objects == {}


async def test_retry_only_for_failed(client, sample_pdf_bytes) -> None:
    document_id = (await _upload(client, sample_pdf_bytes)).json()["document_id"]
    response = await client.post(f"/api/v1/documents/{document_id}/retry")
    assert response.status_code == 409


async def test_requeue_document_lost_while_queued(client, sample_pdf_bytes, job_runner) -> None:
    job_runner.run_jobs = False  # simulate a job lost by a process restart
    document_id = (await _upload(client, sample_pdf_bytes)).json()["document_id"]
    job_runner.run_jobs = True
    response = await client.post(f"/api/v1/documents/{document_id}/retry")
    assert response.status_code == 202
    status = (await client.get(f"/api/v1/documents/{document_id}/status")).json()
    assert status["status"] == "ready"


async def test_retry_after_embedding_failure(client, sample_pdf_bytes, embedder) -> None:
    embedder.fail_times = 1
    document_id = (await _upload(client, sample_pdf_bytes)).json()["document_id"]
    status = (await client.get(f"/api/v1/documents/{document_id}/status")).json()
    assert status["status"] == "failed"
    assert "rate limit" in status["error_message"]

    response = await client.post(f"/api/v1/documents/{document_id}/retry")
    assert response.status_code == 202
    status = (await client.get(f"/api/v1/documents/{document_id}/status")).json()
    assert status["status"] == "ready"
    assert status["error_message"] is None
    assert status["ingestion_attempts"] == 2
