"""Document upload, retry, and deletion workflows (owner-scoped).

PostgreSQL and Supabase Storage are separate systems, so none of these operations
is atomic across both. Ordering is chosen so failures leave recoverable state:

* upload: storage first, then the DB row; if the row insert fails the object is
  deleted (best effort; the orphaned path is logged if that also fails).
* delete: mark ``deleting`` (committed) -> delete object -> delete row. If storage
  deletion fails the row stays ``deleting`` and a repeated DELETE resumes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import BinaryIO

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import (
    AppError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    ValidationFailedError,
)
from app.db.models import Document, DocumentStatus, DocumentVisibility
from app.dependencies.auth import Principal
from app.repositories.document_repository import DocumentRepository
from app.services.document_extractor import inspect_document
from app.services.file_types import (
    FileKind,
    ResolvedFileType,
    has_magic,
    invalid_content_code,
    resolve_upload_type,
)
from app.services.storage_service import StorageService, build_storage_path

logger = logging.getLogger(__name__)

_READ_CHUNK = 1024 * 1024


@dataclass(slots=True)
class IncomingFile:
    filename: str | None
    content_type: str | None
    stream: BinaryIO


@dataclass(slots=True)
class UploadResult:
    document: Document
    duplicate: bool


@dataclass(slots=True)
class _SpooledFile:
    path: Path
    size: int
    sha256: str
    head: bytes
    file_type: ResolvedFileType


def _clean_filename(filename: str | None) -> str:
    name = PurePath((filename or "").replace("\\", "/")).name.strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:255]


def _copy_and_hash(
    source: BinaryIO, destination: BinaryIO, max_bytes: int
) -> tuple[int, str, bytes]:
    """Single pass: copy to disk, hash, and enforce the size limit."""
    digest = hashlib.sha256()
    size = 0
    head = b""
    source.seek(0)
    while chunk := source.read(_READ_CHUNK):
        size += len(chunk)
        if size > max_bytes:
            raise PayloadTooLargeError(
                f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB."
            )
        if len(head) < 1024:
            head += chunk[: 1024 - len(head)]
        digest.update(chunk)
        destination.write(chunk)
    destination.flush()
    return size, digest.hexdigest(), head


class DocumentService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: StorageService | None,
        *,
        max_upload_bytes: int,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._max_upload_bytes = max_upload_bytes

    def _require_storage(self) -> StorageService:
        if self._storage is None:
            raise ServiceUnavailableError("File storage is not configured.")
        return self._storage

    # -- upload ----------------------------------------------------------------

    def _validate_metadata(self, incoming: IncomingFile) -> tuple[str, ResolvedFileType]:
        filename = _clean_filename(incoming.filename)
        if not filename:
            raise ValidationFailedError("A file is required.")
        file_type = resolve_upload_type(filename, incoming.content_type)
        return filename, file_type

    @staticmethod
    def _dedupe_result(existing: Document, visibility: DocumentVisibility) -> UploadResult:
        if existing.status == DocumentStatus.DELETING.value:
            raise ConflictError("This file is currently being deleted; try again shortly.")
        if existing.visibility != visibility.value:
            raise ConflictError(
                f"This file was already uploaded with visibility '{existing.visibility}'.",
                code="duplicate_visibility_mismatch",
            )
        return UploadResult(document=existing, duplicate=True)

    async def upload(
        self,
        principal: Principal,
        incoming: IncomingFile,
        *,
        visibility: DocumentVisibility,
        title: str | None,
    ) -> UploadResult:
        filename, file_type = self._validate_metadata(incoming)
        storage = self._require_storage()
        invalid_code = invalid_content_code(file_type.kind)
        kind_label = "PDF" if file_type.kind is FileKind.PDF else "image"

        with tempfile.NamedTemporaryFile(prefix="aitwin-upload-", suffix=file_type.suffix) as tmp:
            size, sha256, head = await asyncio.to_thread(
                _copy_and_hash, incoming.stream, tmp, self._max_upload_bytes
            )
            spooled = _SpooledFile(Path(tmp.name), size, sha256, head, file_type)
            if spooled.size == 0:
                raise ValidationFailedError("The uploaded file is empty.", code=invalid_code)
            if not has_magic(file_type.kind, spooled.head):
                raise ValidationFailedError(
                    f"The file does not look like a {kind_label}.", code=invalid_code
                )
            try:
                await asyncio.to_thread(inspect_document, spooled.path, file_type.kind)
            except AppError as exc:
                raise ValidationFailedError(exc.message, code=invalid_code) from exc

            async with self._session_factory() as session:
                existing = await DocumentRepository(session).find_by_hash(
                    principal.owner_id, spooled.sha256
                )
            if existing is not None:
                return self._dedupe_result(existing, visibility)

            document_id = uuid.uuid4()
            storage_path = build_storage_path(principal.owner_id, document_id, file_type.suffix)
            tmp.seek(0)
            await storage.upload(storage_path, tmp, spooled.size, file_type.mime_type)

        clean_title = " ".join((title or "").split())[:255] or PurePath(filename).stem[:255]
        document = Document(
            id=document_id,
            owner_id=principal.owner_id,
            title=clean_title or filename,
            original_filename=filename,
            storage_path=storage_path,
            mime_type=file_type.mime_type,
            file_size_bytes=spooled.size,
            file_hash=spooled.sha256,
            visibility=visibility.value,
            status=DocumentStatus.UPLOADED.value,
        )
        try:
            async with self._session_factory() as session, session.begin():
                DocumentRepository(session).add(document)
        except IntegrityError:
            # Lost a race with a concurrent identical upload: keep theirs, drop ours.
            await self._discard_object(storage_path, document_id)
            async with self._session_factory() as session:
                existing = await DocumentRepository(session).find_by_hash(
                    principal.owner_id, spooled.sha256
                )
            if existing is None:
                raise ServiceUnavailableError(
                    "Could not save the document; please retry."
                ) from None
            return self._dedupe_result(existing, visibility)
        except SQLAlchemyError as exc:
            logger.warning(
                "document_insert_failed",
                extra={"document_id": str(document_id), "exc_type": type(exc).__name__},
            )
            await self._discard_object(storage_path, document_id)
            raise ServiceUnavailableError("Could not save the document; please retry.") from exc

        logger.info(
            "document_uploaded",
            extra={"document_id": str(document_id), "bytes": spooled.size},
        )
        return UploadResult(document=document, duplicate=False)

    async def _discard_object(self, storage_path: str, document_id: uuid.UUID) -> None:
        try:
            await self._require_storage().delete(storage_path)
        except Exception:
            logger.error(
                "orphaned_storage_object",
                extra={"document_id": str(document_id), "storage_path": storage_path},
            )

    # -- retry -----------------------------------------------------------------

    async def prepare_retry(
        self, principal: Principal, document_id: uuid.UUID, stale_before: datetime
    ) -> Document:
        async with self._session_factory() as session, session.begin():
            repo = DocumentRepository(session)
            if not await repo.reset_for_retry(document_id, principal.owner_id, stale_before):
                document = await repo.get_for_owner(document_id, principal.owner_id)
                if document is None:
                    raise NotFoundError("Document not found.")
                raise ConflictError(
                    "Only failed, queued, or stalled documents can be retried "
                    f"(current status: {document.status})."
                )
        async with self._session_factory() as session:
            document = await DocumentRepository(session).get_for_owner(
                document_id, principal.owner_id
            )
        if document is None:
            raise NotFoundError("Document not found.")
        return document

    # -- delete ----------------------------------------------------------------

    async def delete(self, principal: Principal, document_id: uuid.UUID) -> None:
        storage = self._require_storage()
        async with self._session_factory() as session, session.begin():
            document = await DocumentRepository(session).mark_deleting(
                document_id, principal.owner_id
            )
            if document is None:
                raise NotFoundError("Document not found.")
            storage_path = document.storage_path

        log = {"document_id": str(document_id)}
        logger.info("document_deleting", extra=log)
        await storage.delete(storage_path)  # StorageError -> 502; row stays 'deleting'

        async with self._session_factory() as session, session.begin():
            await DocumentRepository(session).delete(document_id)
        logger.info("document_deleted", extra=log)
