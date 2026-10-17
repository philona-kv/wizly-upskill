from __future__ import annotations

import uuid

import pytest

from app.db.models import Document, DocumentChunk
from app.db.models.document_chunk import EMBEDDING_DIMENSIONS
from tests.db.conftest import OWNER_A, OWNER_B

pytestmark = pytest.mark.db

UNIT = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


async def _ready_document(
    session_factory,
    *,
    owner_id=OWNER_A,
    visibility="global",
    status="ready",
    content="Cancellations must be requested at least 48 hours in advance.",
    model="text-embedding-3-small",
    title="Cancellation Policy",
    filename="cancellation_policy.pdf",
):
    document_id = uuid.uuid4()
    async with session_factory() as session, session.begin():
        session.add(
            Document(
                id=document_id,
                owner_id=owner_id,
                title=title,
                original_filename=filename,
                storage_path=f"{owner_id}/{document_id}.pdf",
                mime_type="application/pdf",
                file_size_bytes=100,
                file_hash=uuid.uuid4().hex,
                visibility=visibility,
                status=status,
                chunk_count=1,
                embedding_model=model,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
            )
        )
        session.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=0,
                content=content,
                page_start=4,
                page_end=4,
                token_count=20,
                embedding=list(UNIT),
            )
        )
    return document_id


async def test_create_list_get_delete_conversation(client) -> None:
    created = await client.post("/api/v1/conversations", json={})
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    assert created.json()["title"] == "New conversation"

    listed = await client.get("/api/v1/conversations")
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == conversation_id

    fetched = await client.get(f"/api/v1/conversations/{conversation_id}")
    assert fetched.status_code == 200

    deleted = await client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/v1/conversations/{conversation_id}")).status_code == 404


async def test_conversation_ownership(client, current_owner) -> None:
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    current_owner.owner_id = OWNER_B
    assert (await client.get(f"/api/v1/conversations/{conversation_id}")).status_code == 404
    messages_url = f"/api/v1/conversations/{conversation_id}/messages"
    assert (await client.get(messages_url)).status_code == 404
    assert (
        await client.post(
            f"/api/v1/conversations/{conversation_id}/messages", json={"content": "hi"}
        )
    ).status_code == 404
    assert (await client.delete(f"/api/v1/conversations/{conversation_id}")).status_code == 404


async def test_empty_message_rejected(client) -> None:
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "   "}
    )
    assert response.status_code == 422


async def test_chat_turn_persists_citations_and_reloads(
    client, session_factory, generator
) -> None:
    await _ready_document(session_factory)
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "What is the cancellation policy?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_message"]["role"] == "user"
    assert "[S1]" in body["assistant_message"]["content"]
    citations = body["assistant_message"]["citations"]
    assert citations[0]["page_start"] == 4
    assert citations[0]["original_filename"] == "cancellation_policy.pdf"
    assert citations[0]["excerpt"]
    assert generator.calls == 1
    assert "untrusted" in generator.last_evidence.lower()

    history = await client.get(f"/api/v1/conversations/{conversation_id}/messages")
    items = history.json()["items"]
    assert [m["role"] for m in items] == ["user", "assistant"]
    assert items[1]["citations"][0]["citation_id"] == "S1"

    listed = await client.get("/api/v1/conversations")
    assert listed.json()["items"][0]["title"].startswith("What is the cancellation")


async def test_no_documents_abstains_without_llm(client, generator) -> None:
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    body = (
        await client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "What is the cancellation policy?"},
        )
    ).json()
    assert "enough evidence" in body["assistant_message"]["content"].lower()
    assert body["assistant_message"]["citations"] == []
    assert generator.calls == 0


async def test_private_and_other_owner_documents_never_retrieved(
    client, session_factory, generator
) -> None:
    await _ready_document(
        session_factory, visibility="private", content="SECRET private playground text 48 hours."
    )
    await _ready_document(
        session_factory,
        owner_id=OWNER_B,
        content="Other owner's cancellation is 99 days.",
        filename="other.pdf",
    )
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    body = (
        await client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "What is the cancellation policy?"},
        )
    ).json()
    assert generator.calls == 0
    text = body["assistant_message"]["content"].lower()
    assert "secret" not in text
    assert "99 days" not in text
    assert body["assistant_message"]["citations"] == []


async def test_non_ready_documents_excluded(client, session_factory, generator) -> None:
    await _ready_document(session_factory, status="processing")
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    body = (
        await client.post(
            f"/api/v1/conversations/{conversation_id}/messages", json={"content": "policy?"}
        )
    ).json()
    assert generator.calls == 0
    assert "enough evidence" in body["assistant_message"]["content"].lower()


async def test_generation_failure_keeps_user_message(client, session_factory, generator) -> None:
    await _ready_document(session_factory)
    generator.fail = True
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "cancellation?"}
    )
    assert response.status_code == 502
    history = (await client.get(f"/api/v1/conversations/{conversation_id}/messages")).json()
    assert history["total"] == 1
    assert history["items"][0]["role"] == "user"


async def test_idempotent_client_message_id(client, session_factory, generator) -> None:
    await _ready_document(session_factory)
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    payload = {
        "content": "What is the cancellation policy?",
        "client_message_id": str(uuid.uuid4()),
    }
    first = await client.post(f"/api/v1/conversations/{conversation_id}/messages", json=payload)
    second = await client.post(f"/api/v1/conversations/{conversation_id}/messages", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["user_message"]["id"] == second.json()["user_message"]["id"]
    assert first.json()["assistant_message"]["id"] == second.json()["assistant_message"]["id"]
    history = (await client.get(f"/api/v1/conversations/{conversation_id}/messages")).json()
    assert history["total"] == 2
    assert generator.calls == 1


async def test_embedding_failure_is_not_empty_answer(
    client, session_factory, embedder
) -> None:
    await _ready_document(session_factory)
    embedder.fail_times = 5
    conversation_id = (await client.post("/api/v1/conversations", json={})).json()["id"]
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json={"content": "policy?"}
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "embedding_error"
    history = (await client.get(f"/api/v1/conversations/{conversation_id}/messages")).json()
    assert history["items"][0]["role"] == "user"
    assert len(history["items"]) == 1
