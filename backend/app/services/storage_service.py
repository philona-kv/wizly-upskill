"""Original-file storage (Supabase Storage via its REST API).

Uses the service-role key, so it must only ever run on the backend.
Object paths are ``{owner_id}/{document_id}.{ext}`` where ``ext`` is
``pdf``, ``png``, or ``jpg`` - no user-provided filenames.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import BinaryIO, Protocol

import httpx

from app.core.config import Settings
from app.core.exceptions import StorageError, StorageObjectNotFoundError

logger = logging.getLogger(__name__)

_STREAM_CHUNK = 1024 * 1024


def build_storage_path(owner_id: uuid.UUID, document_id: uuid.UUID, suffix: str = ".pdf") -> str:
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{owner_id}/{document_id}{ext}"


class StorageService(Protocol):
    async def upload(self, path: str, source: BinaryIO, size: int, content_type: str) -> None: ...

    async def download_to(self, path: str, destination: BinaryIO) -> int: ...

    async def delete(self, path: str) -> None:
        """Delete an object. Deleting a missing object is not an error."""
        ...


async def _iter_file(source: BinaryIO) -> AsyncIterator[bytes]:
    source.seek(0)
    while chunk := source.read(_STREAM_CHUNK):
        yield chunk


class SupabaseStorageService:
    def __init__(self, client: httpx.AsyncClient, base_url: str, service_key: str, bucket: str):
        self._client = client
        self._base = f"{base_url.rstrip('/')}/storage/v1"
        self._bucket = bucket
        self._headers = {"Authorization": f"Bearer {service_key}", "apikey": service_key}

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient) -> SupabaseStorageService:
        url, key = settings.require_supabase()
        return cls(client, url, key, settings.supabase_storage_bucket)

    def _object_url(self, path: str) -> str:
        return f"{self._base}/object/{self._bucket}/{path}"

    async def upload(self, path: str, source: BinaryIO, size: int, content_type: str) -> None:
        headers = {
            **self._headers,
            "Content-Type": content_type,
            "Content-Length": str(size),
            "x-upsert": "false",
            "cache-control": "no-store",
        }
        try:
            response = await self._client.post(
                self._object_url(path), content=_iter_file(source), headers=headers
            )
        except httpx.HTTPError as exc:
            logger.warning("storage_upload_transport_error", extra={"exc_type": type(exc).__name__})
            raise StorageError("Could not reach file storage.") from exc
        if response.is_error:
            logger.warning("storage_upload_failed", extra={"http_status": response.status_code})
            raise StorageError("File storage rejected the upload.")

    async def download_to(self, path: str, destination: BinaryIO) -> int:
        written = 0
        try:
            async with self._client.stream(
                "GET", self._object_url(path), headers=self._headers
            ) as response:
                if response.status_code in (400, 404):
                    # Supabase reports missing objects as 400 {"error": "not_found"} or 404.
                    raise StorageObjectNotFoundError()
                if response.is_error:
                    logger.warning(
                        "storage_download_failed", extra={"http_status": response.status_code}
                    )
                    raise StorageError("Could not download the stored file.")
                async for chunk in response.aiter_bytes(_STREAM_CHUNK):
                    destination.write(chunk)
                    written += len(chunk)
        except httpx.HTTPError as exc:
            logger.warning(
                "storage_download_transport_error", extra={"exc_type": type(exc).__name__}
            )
            raise StorageError("Could not reach file storage.") from exc
        destination.flush()
        return written

    async def delete(self, path: str) -> None:
        # Bulk-delete endpoint: returns 200 with the removed objects, and 200 with an empty
        # list when the object does not exist, which makes deletion idempotent.
        try:
            response = await self._client.request(
                "DELETE",
                f"{self._base}/object/{self._bucket}",
                json={"prefixes": [path]},
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            logger.warning("storage_delete_transport_error", extra={"exc_type": type(exc).__name__})
            raise StorageError("Could not reach file storage.") from exc
        if response.status_code == 404:
            return
        if response.is_error:
            logger.warning("storage_delete_failed", extra={"http_status": response.status_code})
            raise StorageError("Could not delete the stored file.")
