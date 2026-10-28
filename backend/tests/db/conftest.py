"""Fixtures for tests that need a real PostgreSQL + pgvector database.

Set TEST_DATABASE_URL to a disposable database, e.g.
    postgresql://aitwin:aitwin@localhost:54329/aitwin_test        (docker compose)
The schema is rebuilt with Alembic (downgrade base -> upgrade head) once per session
and tables are truncated between tests. A guard refuses non-local hosts and database
names without "test" in them.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings, to_async_database_url
from app.core.exceptions import (
    EmbeddingError,
    GenerationError,
    StorageError,
    StorageObjectNotFoundError,
)
from app.db.models import EMBEDDING_DIMENSIONS
from app.db.session import create_engine, create_session_factory
from app.dependencies.auth import Principal, get_current_principal
from app.dependencies.services import get_ingestion_service, get_job_runner
from app.main import create_app
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService

BACKEND_DIR = Path(__file__).resolve().parents[2]
OWNER_A = uuid.UUID("00000000-0000-0000-0000-00000000000a")
OWNER_B = uuid.UUID("00000000-0000-0000-0000-00000000000b")

pytestmark = pytest.mark.db


def _test_database_url() -> str:
    raw = os.environ.get("TEST_DATABASE_URL", "")
    if not raw:
        pytest.skip("TEST_DATABASE_URL is not set; skipping database tests")
    url = make_url(to_async_database_url(raw))
    host = url.host or url.query.get("host", "")
    if isinstance(host, tuple):
        host = host[0]
    is_local = host in {"localhost", "127.0.0.1", "::1"} or str(host).startswith("/")
    if not is_local or "test" not in (url.database or ""):
        raise RuntimeError(
            "Refusing to run DB tests: TEST_DATABASE_URL must point at a local database "
            "whose name contains 'test'."
        )
    return url.render_as_string(hide_password=False)


def _alembic_config(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.cmd_opts = type("Opts", (), {"x": [f"db_url={url}"]})()
    return cfg


@pytest.fixture(scope="session")
def database_url() -> str:
    return _test_database_url()


@pytest.fixture(scope="session")
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    cfg = _alembic_config(database_url)
    # env.py calls asyncio.run(), so migrations run in a worker thread.
    await asyncio.to_thread(command.downgrade, cfg, "base")
    await asyncio.to_thread(command.upgrade, cfg, "head")
    settings = Settings(_env_file=None, app_env="test", database_url=database_url)
    eng = create_engine(settings)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session_factory(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE playground_experiment_runs, playground_experiments, "
                "prompt_template_versions, prompt_templates, document_promotion_audits, "
                "workflow_executions, owner_memories, conversation_summaries, "
                "messages, conversations, document_chunks, documents CASCADE"
            )
        )
    yield create_session_factory(engine)


# -- fakes -------------------------------------------------------------------


@dataclass
class FakeStorage:
    objects: dict[str, bytes] = field(default_factory=dict)
    fail_upload: bool = False
    fail_download: bool = False
    fail_delete: bool = False
    deleted: list[str] = field(default_factory=list)

    async def upload(self, path: str, source: BinaryIO, size: int, content_type: str) -> None:
        if self.fail_upload:
            raise StorageError("File storage rejected the upload.")
        if path in self.objects:
            raise StorageError("Object already exists.")
        source.seek(0)
        self.objects[path] = source.read()
        assert len(self.objects[path]) == size

    async def download_to(self, path: str, destination: BinaryIO) -> int:
        if self.fail_download:
            raise StorageError("Could not reach file storage.")
        if path not in self.objects:
            raise StorageObjectNotFoundError()
        destination.write(self.objects[path])
        destination.flush()
        return len(self.objects[path])

    async def delete(self, path: str) -> None:
        if self.fail_delete:
            raise StorageError("Could not delete the stored file.")
        self.objects.pop(path, None)
        self.deleted.append(path)


@dataclass
class FakeGenerator:
    model: str = "gpt-6.1-sol"
    fail: bool = False
    calls: int = 0
    last_evidence: str = ""
    last_question: str = ""
    response: str = "The policy requires 48 hours notice. [S1]"

    async def generate(
        self, *, question: str, evidence_block: str, history, instructions=None
    ) -> str:
        self.calls += 1
        self.last_question = question
        self.last_evidence = evidence_block
        if self.fail:
            raise GenerationError("The language model timed out.")
        if "S1" in evidence_block:
            return self.response
        return "I cannot answer from the supplied evidence."


@dataclass
class FakeEmbedder:
    model: str = "text-embedding-3-small"
    dimensions: int = EMBEDDING_DIMENSIONS
    fail_times: int = 0
    calls: int = 0
    before_return: object = None  # optional async hook, used for race tests

    async def embed(
        self, texts: Sequence[str], token_counts: Sequence[int] | None = None
    ) -> list[list[float]]:
        self.calls += 1
        if self.before_return is not None:
            await self.before_return()
        if self.fail_times > 0:
            self.fail_times -= 1
            raise EmbeddingError(
                "Embedding provider rate limit or quota exceeded; try again later."
            )
        return [[(i + 1) / 1000.0] + [0.0] * (self.dimensions - 1) for i in range(len(texts))]


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def ingestion(session_factory, storage, embedder) -> IngestionService:
    return IngestionService(
        session_factory,
        storage,
        ChunkingService(120, 20, embedder),
        embedder,
        stale_after_seconds=900,
    )


class InlineJobRunner:
    """Runs ingestion immediately (still after the upload row is committed)."""

    def __init__(self, ingestion: IngestionService) -> None:
        self.ingestion = ingestion
        self.enqueued: list[uuid.UUID] = []
        self.run_jobs = True

    async def enqueue(self, document_id: uuid.UUID) -> None:
        self.enqueued.append(document_id)
        if self.run_jobs:
            await self.ingestion.run(document_id)


@dataclass
class CurrentOwner:
    owner_id: uuid.UUID = OWNER_A


@pytest.fixture
def current_owner() -> CurrentOwner:
    return CurrentOwner()


@pytest.fixture
def job_runner(ingestion) -> InlineJobRunner:
    return InlineJobRunner(ingestion)


@pytest.fixture
async def client(
    session_factory, storage, embedder, generator, ingestion, job_runner, current_owner
) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(_env_file=None, app_env="test", max_upload_size_mb=1, log_level="WARNING")
    app = create_app(settings, use_lifespan=False)
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.embedder = embedder
    app.state.generator = generator
    app.dependency_overrides[get_current_principal] = lambda: Principal(current_owner.owner_id)
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_job_runner] = lambda: job_runner
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
