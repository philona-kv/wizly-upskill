from __future__ import annotations

import json

import httpx
import pytest

from services.api_client import ApiClient, ApiError

DOC = {
    "id": "8a7c4f0e-1111-2222-3333-444455556666",
    "title": "Report",
    "original_filename": "report.pdf",
    "mime_type": "application/pdf",
    "file_size_bytes": 2048,
    "file_hash": "abc",
    "visibility": "global",
    "status": "processing",
    "error_message": None,
    "page_count": None,
    "chunk_count": 0,
    "embedding_model": None,
    "ingestion_attempts": 1,
    "created_at": "2026-10-12T10:00:00Z",
    "updated_at": "2026-10-12T10:00:01+00:00",
    "processed_at": None,
}


def client_for(handler) -> ApiClient:
    return ApiClient("http://backend:8000/", transport=httpx.MockTransport(handler))


def test_list_documents_parses_and_sends_filters() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(200, json={"items": [DOC], "total": 1, "limit": 5, "offset": 0})

    page = client_for(handler).list_documents(limit=5, status="ready", visibility=None)
    assert seen["url"].path == "/api/v1/documents"
    assert dict(seen["url"].params) == {"limit": "5", "offset": "0", "status": "ready"}
    assert page.total == 1
    doc = page.items[0]
    assert doc.title == "Report" and doc.status == "processing" and not doc.is_terminal
    assert doc.created_at is not None and doc.created_at.tzinfo is not None


def test_backend_error_envelope_is_surfaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            415, json={"error": {"code": "unsupported_media_type", "message": "Only PDFs."}}
        )

    with pytest.raises(ApiError) as info:
        client_for(handler).upload_document("a.txt", b"x", visibility="global")
    assert info.value.status_code == 415
    assert info.value.code == "unsupported_media_type"
    assert info.value.message == "Only PDFs."


def test_non_json_error_gets_generic_message() -> None:
    with pytest.raises(ApiError) as info:
        client_for(lambda r: httpx.Response(500, text="<html>oops</html>")).get_document("x")
    assert info.value.status_code == 500
    assert "500" in info.value.message


def test_backend_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError) as info:
        client_for(handler).health()
    assert info.value.is_unavailable


def test_timeout_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiError, match="in time"):
        client_for(handler).list_documents()


def test_invalid_payload_shape() -> None:
    with pytest.raises(ApiError) as info:
        client_for(lambda r: httpx.Response(200, json={"items": [{"id": 1}]})).list_documents()
    assert info.value.code == "invalid_response"


def test_upload_sends_multipart_with_title() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen["body"] = body
        seen["ctype"] = request.headers["content-type"]
        return httpx.Response(
            202,
            json={
                "document_id": DOC["id"],
                "status": "uploaded",
                "message": "accepted",
                "duplicate": False,
                "document": DOC,
            },
        )

    result = client_for(handler).upload_document(
        "r.pdf", b"%PDF-1.4", visibility="private", title="  My title "
    )
    assert seen["ctype"].startswith("multipart/form-data")
    assert b'name="visibility"' in seen["body"] and b"private" in seen["body"]
    assert b"My title" in seen["body"] and b"%PDF-1.4" in seen["body"]
    assert result.duplicate is False and result.message == "accepted"


def test_upload_image_sends_image_content_type() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(
            202,
            json={
                "document_id": DOC["id"],
                "status": "uploaded",
                "message": "accepted",
                "duplicate": False,
                "document": {**DOC, "original_filename": "scan.png", "mime_type": "image/png"},
            },
        )

    client_for(handler).upload_document("scan.png", b"\x89PNG", visibility="global")
    assert b"image/png" in seen["body"]


def test_status_and_delete() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "id": DOC["id"],
                    "status": "failed",
                    "error_message": "No text",
                    "page_count": 2,
                    "chunk_count": 0,
                }
            ),
        )

    client = client_for(handler)
    status = client.get_status(DOC["id"])
    assert status.is_terminal and status.error_message == "No text"
    client.delete_document(DOC["id"])
    assert calls[-1] == ("DELETE", f"/api/v1/documents/{DOC['id']}")


def test_readiness_503_is_not_an_exception() -> None:
    client = client_for(lambda r: httpx.Response(503, json={"status": "not_ready", "checks": {}}))
    assert client.readiness()["status"] == "not_ready"


TURN = {
    "conversation_id": "11111111-1111-1111-1111-111111111111",
    "user_message": {
        "id": "22222222-2222-2222-2222-222222222222",
        "role": "user",
        "content": "What is the cancellation policy?",
        "citations": None,
        "model_name": None,
        "client_message_id": "33333333-3333-3333-3333-333333333333",
        "created_at": "2026-10-17T10:00:00Z",
    },
    "assistant_message": {
        "id": "44444444-4444-4444-4444-444444444444",
        "role": "assistant",
        "content": "Cancellations need 48 hours notice. [S1]",
        "citations": [
            {
                "citation_id": "S1",
                "document_id": DOC["id"],
                "document_title": "Cancellation Policy",
                "original_filename": "cancellation_policy.pdf",
                "page_start": 4,
                "page_end": 4,
                "chunk_id": "55555555-5555-5555-5555-555555555555",
                "excerpt": "Cancellations must be requested at least 48 hours in advance.",
            }
        ],
        "model_name": "gpt-6.1-sol",
        "client_message_id": None,
        "created_at": "2026-10-17T10:00:01Z",
    },
}


def test_send_message_parses_citations_and_sends_client_id() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=TURN)

    turn = client_for(handler).send_message(
        TURN["conversation_id"],
        "What is the cancellation policy?",
        client_message_id="33333333-3333-3333-3333-333333333333",
    )
    assert seen["url"].path == f"/api/v1/conversations/{TURN['conversation_id']}/messages"
    assert seen["body"]["client_message_id"] == "33333333-3333-3333-3333-333333333333"
    assert turn.assistant.citations[0].citation_id == "S1"
    assert turn.assistant.citations[0].page_start == 4
    assert "48 hours" in (turn.assistant.citations[0].excerpt or "")
    assert turn.user.content.startswith("What is")


def test_send_message_surfaces_generation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={
                "error": {
                    "code": "generation_error",
                    "message": "The language model timed out.",
                }
            },
        )

    with pytest.raises(ApiError) as info:
        client_for(handler).send_message("c", "hello")
    assert info.value.status_code == 502
    assert info.value.code == "generation_error"
    assert "timed out" in info.value.message


def test_list_conversations_and_messages() -> None:
    conv = {
        "id": TURN["conversation_id"],
        "title": "What is the cancellation policy?",
        "created_at": "2026-10-17T10:00:00Z",
        "updated_at": "2026-10-17T10:00:01Z",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages"):
            return httpx.Response(
                200,
                json={
                    "items": [TURN["user_message"], TURN["assistant_message"]],
                    "total": 2,
                    "limit": 100,
                    "offset": 0,
                },
            )
        return httpx.Response(200, json={"items": [conv], "total": 1, "limit": 50, "offset": 0})

    api = client_for(handler)
    listed = api.list_conversations()
    assert listed.total == 1 and listed.items[0].title.startswith("What is")
    history = api.get_messages(TURN["conversation_id"])
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].citations[0].original_filename == "cancellation_policy.pdf"
