"""FastAPI application factory.

Run locally from ``backend/``: ``uvicorn app.main:app --reload``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.api.router import api_router
from app.core.config import AppEnv, ConfigurationError, Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db.models import EMBEDDING_DIMENSIONS
from app.db.session import create_engine, create_session_factory
from app.services.embedding_service import OpenAIEmbeddingService
from app.services.generation_service import OpenAIGenerationService
from app.services.storage_service import SupabaseStorageService
from app.workflows.checkpointer import open_postgres_checkpointer

logger = logging.getLogger(__name__)

_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _check_embedding_dimensions(settings: Settings) -> None:
    if settings.openai_embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise ConfigurationError(
            f"OPENAI_EMBEDDING_DIMENSIONS={settings.openai_embedding_dimensions} does not match "
            f"the database schema (vector({EMBEDDING_DIMENSIONS})). Changing it requires a "
            "migration and re-embedding all documents."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    strict = settings.app_env == AppEnv.PRODUCTION
    _check_embedding_dimensions(settings)

    engine = None
    if settings.database_url:
        engine = create_engine(settings)
        app.state.session_factory = create_session_factory(engine)
    elif strict:
        raise ConfigurationError("DATABASE_URL is not configured")
    else:
        logger.warning("startup_missing_config", extra={"component": "database"})

    storage_http: httpx.AsyncClient | None = httpx.AsyncClient(
        timeout=settings.storage_timeout_seconds
    )
    try:
        app.state.storage = SupabaseStorageService.from_settings(settings, storage_http)
    except ConfigurationError:
        await storage_http.aclose()
        storage_http = None
        if strict:
            raise
        logger.warning("startup_missing_config", extra={"component": "storage"})

    embedder: OpenAIEmbeddingService | None = None
    llm_client: AsyncOpenAI | None = None
    try:
        embedder = OpenAIEmbeddingService.from_settings(settings)
        app.state.embedder = embedder
        llm_kwargs: dict = {
            "api_key": settings.require_openai(),
            "timeout": settings.openai_timeout_seconds,
            "max_retries": 0,
        }
        base_url = settings.embedding_api_base()
        if base_url is not None:
            llm_kwargs["base_url"] = base_url
        llm_client = AsyncOpenAI(**llm_kwargs)
        app.state.generator = OpenAIGenerationService.from_settings(settings, llm_client)
    except ConfigurationError:
        if strict:
            raise
        logger.warning("startup_missing_config", extra={"component": "embeddings"})

    app.state.langgraph_checkpointer = None
    logger.info("startup_complete", extra={"app_env": settings.app_env.value})
    try:
        async with open_postgres_checkpointer(settings) as checkpointer:
            app.state.langgraph_checkpointer = checkpointer
            yield
    finally:
        if llm_client is not None:
            await llm_client.close()
        if embedder is not None:
            await embedder.aclose()
        if storage_http is not None:
            await storage_http.aclose()
        if engine is not None:
            await engine.dispose()


def create_app(settings: Settings | None = None, *, use_lifespan: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="AI Twin API",
        version="0.1.0",
        lifespan=lifespan if use_lifespan else None,
        docs_url="/docs" if settings.app_env != AppEnv.PRODUCTION else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.session_factory = None
    app.state.storage = None
    app.state.embedder = None
    app.state.generator = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)

    # Multipart bodies are parsed (and spooled to disk) before route code runs, so reject
    # obviously oversized uploads from Content-Length up front. Chunked uploads without
    # Content-Length are still capped by the streaming size check in DocumentService.
    upload_limit = settings.max_upload_size_bytes + _MULTIPART_OVERHEAD_BYTES

    @app.middleware("http")
    async def _limit_upload_size(request: Request, call_next):
        if request.method == "POST" and request.url.path.rstrip("/") == "/api/v1/documents":
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > upload_limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "payload_too_large",
                            "message": "File exceeds the maximum upload size of "
                            f"{settings.max_upload_size_mb} MB.",
                        }
                    },
                )
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": "AI Twin API",
            "health": "/api/v1/health",
            "readiness": "/api/v1/health/ready",
            "docs": "/docs",
        }

    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
