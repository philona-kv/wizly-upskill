import io
import json
import uuid

import httpx
import pytest

from app.core.exceptions import StorageError, StorageObjectNotFoundError
from app.services.storage_service import SupabaseStorageService, build_storage_path

KEY = "service-role-key"


def _service(handler) -> tuple[SupabaseStorageService, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SupabaseStorageService(client, "https://proj.supabase.co/", KEY, "documents"), client


def test_storage_path_has_no_user_filename() -> None:
    owner, doc = uuid.uuid4(), uuid.uuid4()
    assert build_storage_path(owner, doc) == f"{owner}/{doc}.pdf"
    assert build_storage_path(owner, doc, ".png") == f"{owner}/{doc}.png"
    assert build_storage_path(owner, doc, ".jpg") == f"{owner}/{doc}.jpg"


async def test_upload_sends_auth_and_content() -> None:
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["upsert"] = request.headers["x-upsert"]
        seen["body"] = await request.aread()
        return httpx.Response(200, json={"Key": "documents/x"})

    service, client = _service(handler)
    await service.upload("o/d.pdf", io.BytesIO(b"%PDF-data"), 9, "application/pdf")
    await client.aclose()
    assert seen["url"] == "https://proj.supabase.co/storage/v1/object/documents/o/d.pdf"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["upsert"] == "false"
    assert seen["body"] == b"%PDF-data"


async def test_upload_error_is_sanitized() -> None:
    service, client = _service(lambda r: httpx.Response(500, text=f"boom {KEY}"))
    with pytest.raises(StorageError) as info:
        await service.upload("o/d.pdf", io.BytesIO(b"x"), 1, "application/pdf")
    await client.aclose()
    assert KEY not in info.value.message


async def test_upload_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    service, client = _service(handler)
    with pytest.raises(StorageError):
        await service.upload("o/d.pdf", io.BytesIO(b"x"), 1, "application/pdf")
    await client.aclose()


async def test_download_streams_to_file() -> None:
    service, client = _service(lambda r: httpx.Response(200, content=b"%PDF-content"))
    buffer = io.BytesIO()
    assert await service.download_to("o/d.pdf", buffer) == len(b"%PDF-content")
    await client.aclose()
    assert buffer.getvalue() == b"%PDF-content"


@pytest.mark.parametrize("status", [400, 404])
async def test_download_missing_object(status: int) -> None:
    service, client = _service(lambda r: httpx.Response(status, json={"error": "not_found"}))
    with pytest.raises(StorageObjectNotFoundError):
        await service.download_to("o/d.pdf", io.BytesIO())
    await client.aclose()


async def test_delete_uses_bulk_endpoint_and_is_idempotent() -> None:
    bodies = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert str(request.url) == "https://proj.supabase.co/storage/v1/object/documents"
        bodies.append(json.loads(await request.aread()))
        return httpx.Response(200, json=[])

    service, client = _service(handler)
    await service.delete("o/d.pdf")
    await service.delete("o/d.pdf")
    await client.aclose()
    assert bodies == [{"prefixes": ["o/d.pdf"]}] * 2


async def test_delete_failure_raises() -> None:
    service, client = _service(lambda r: httpx.Response(503))
    with pytest.raises(StorageError):
        await service.delete("o/d.pdf")
    await client.aclose()
