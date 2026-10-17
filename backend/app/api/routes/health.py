from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()

CheckState = Literal["ok", "error", "not_configured"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_env: str
    development_identity: bool


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckState]


@router.get("", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        app_env=settings.app_env.value,
        development_identity=settings.auth_mode.value == "development",
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    state = request.app.state
    checks: dict[str, CheckState] = {}

    if state.session_factory is None:
        checks["database"] = "not_configured"
        checks["pgvector"] = "not_configured"
    else:
        try:
            async with state.session_factory() as session:
                await session.execute(text("SELECT 1"))
                checks["database"] = "ok"
                has_vector = await session.scalar(
                    text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                )
                checks["pgvector"] = "ok" if has_vector else "error"
        except Exception as exc:
            logger.warning("readiness_db_failed", extra={"exc_type": type(exc).__name__})
            checks.setdefault("database", "error")
            checks.setdefault("pgvector", "error")

    # Storage and embeddings are only checked for configuration; probing them on every
    # readiness call would cost money / quota.
    checks["storage"] = "ok" if state.storage is not None else "not_configured"
    checks["embeddings"] = "ok" if state.embedder is not None else "not_configured"

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
