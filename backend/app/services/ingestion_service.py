"""Ingestion orchestration: storage -> extraction -> chunking -> embeddings -> database.

Transaction boundaries are deliberately short:

1. claim (one UPDATE, committed)
2. download / extract / chunk / embed with NO open transaction
3. finalize (delete old chunks + insert new ones + mark ready, one transaction)
4. on failure: mark failed (separate transaction)

``run`` never raises; outcomes are recorded on the document row.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import AppError, DocumentExtractionError, StorageObjectNotFoundError
from app.repositories.document_repository import DocumentRepository
from app.services.chunking_service import ChunkingService
from app.services.document_extractor import extract_document
from app.services.embedding_service import Embedder
from app.services.file_types import kind_from_storage_path
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)

GENERIC_FAILURE = "Ingestion failed due to an internal error. You can retry."


class IngestionOutcome(StrEnum):
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"  # not eligible / already claimed by another worker
    ABORTED = "aborted"  # document deleted or being deleted mid-run


class IngestionService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: StorageService,
        chunker: ChunkingService,
        embedder: Embedder,
        *,
        stale_after_seconds: int,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._chunker = chunker
        self._embedder = embedder
        self._stale_after = timedelta(seconds=stale_after_seconds)

    def stale_before(self) -> datetime:
        return datetime.now(UTC) - self._stale_after

    async def run(self, document_id: uuid.UUID) -> IngestionOutcome:
        log = {"document_id": str(document_id)}
        try:
            async with self._session_factory() as session, session.begin():
                document = await DocumentRepository(session).claim_for_processing(
                    document_id, self.stale_before()
                )
                if document is not None:
                    storage_path = document.storage_path
                    claimed_at = document.processing_started_at
                    attempt = document.ingestion_attempts
        except Exception:
            logger.exception("ingestion_claim_failed", extra=log)
            return IngestionOutcome.FAILED

        if document is None or claimed_at is None:
            logger.info("ingestion_skipped", extra={**log, "stage": "claim"})
            return IngestionOutcome.SKIPPED

        logger.info("ingestion_started", extra={**log, "attempt": attempt})
        fail = partial(self._fail, document_id, claimed_at)
        stage = "download"
        page_count: int | None = None
        try:
            kind = kind_from_storage_path(storage_path)
            suffix = Path(storage_path).suffix or ".bin"
            with tempfile.NamedTemporaryFile(prefix="aitwin-", suffix=suffix) as tmp:
                size = await self._storage.download_to(storage_path, tmp)
                logger.info("ingestion_stage", extra={**log, "stage": stage, "bytes": size})
                stage = "extract"
                extracted = await asyncio.to_thread(extract_document, Path(tmp.name), kind)
            page_count = extracted.page_count
            logger.info("ingestion_stage", extra={**log, "stage": stage, "pages": page_count})

            stage = "chunk"
            chunks = await asyncio.to_thread(self._chunker.chunk, extracted.pages)
            del extracted
            if not chunks:
                raise DocumentExtractionError(
                    "The document contains no usable text after processing."
                )
            logger.info("ingestion_stage", extra={**log, "stage": stage, "chunks": len(chunks)})

            stage = "embed"
            vectors = await self._embedder.embed(
                [c.content for c in chunks], [c.token_count for c in chunks]
            )
            logger.info("ingestion_stage", extra={**log, "stage": stage})

            stage = "persist"
            async with self._session_factory() as session, session.begin():
                completed = await DocumentRepository(session).complete_ingestion(
                    document_id,
                    claimed_at=claimed_at,
                    chunks=chunks,
                    embeddings=vectors,
                    page_count=page_count,
                    embedding_model=self._embedder.model,
                    embedding_dimensions=self._embedder.dimensions,
                )
        except StorageObjectNotFoundError:
            return await fail("The original file is missing from storage.", stage)
        except AppError as exc:
            return await fail(exc.message, stage, page_count)
        except Exception:
            logger.exception("ingestion_unexpected_error", extra={**log, "stage": stage})
            return await fail(GENERIC_FAILURE, stage, page_count)

        if not completed:
            logger.info("ingestion_aborted", extra={**log, "stage": "persist"})
            return IngestionOutcome.ABORTED
        logger.info("ingestion_ready", extra={**log, "chunks": len(chunks)})
        return IngestionOutcome.READY

    async def _fail(
        self,
        document_id: uuid.UUID,
        claimed_at: datetime,
        message: str,
        stage: str,
        page_count: int | None = None,
    ) -> IngestionOutcome:
        log = {"document_id": str(document_id), "stage": stage}
        logger.warning("ingestion_failed", extra={**log, "reason": message})
        try:
            async with self._session_factory() as session, session.begin():
                updated = await DocumentRepository(session).mark_failed(
                    document_id, message, claimed_at=claimed_at, page_count=page_count
                )
        except Exception:
            # The row stays 'processing'; it becomes reclaimable after the stale timeout.
            logger.exception("ingestion_mark_failed_error", extra=log)
            return IngestionOutcome.FAILED
        return IngestionOutcome.FAILED if updated else IngestionOutcome.ABORTED
