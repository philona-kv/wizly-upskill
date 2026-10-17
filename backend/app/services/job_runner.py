"""Ingestion job scheduling.

``BackgroundTasksJobRunner`` runs ingestion in the API process after the response
is sent. It is NOT durable: a process restart or crash loses queued/in-flight jobs
(the document stays ``uploaded`` or ``processing``; the latter becomes retryable after
``INGESTION_STALE_AFTER_SECONDS``). Replace it with a worker-backed implementation
(ARQ, Celery, ...) whose ``enqueue`` pushes ``document_id`` to a queue and whose
worker calls ``IngestionService.run``. Routes depend only on the protocol.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from fastapi import BackgroundTasks

from app.services.ingestion_service import IngestionService


class IngestionJobRunner(Protocol):
    async def enqueue(self, document_id: uuid.UUID) -> None: ...


class BackgroundTasksJobRunner:
    def __init__(self, background_tasks: BackgroundTasks, ingestion: IngestionService) -> None:
        self._background_tasks = background_tasks
        self._ingestion = ingestion

    async def enqueue(self, document_id: uuid.UUID) -> None:
        self._background_tasks.add_task(self._ingestion.run, document_id)
