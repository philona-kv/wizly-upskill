"""Async engine / session factory construction.

The engine is created once in the FastAPI lifespan and stored on ``app.state``;
there is no module-level global engine.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    url = make_url(settings.require_database())
    connect_args: dict[str, Any] = {"timeout": 10, "command_timeout": 60}
    if settings.database_use_pgbouncer:
        # PgBouncer / Supavisor transaction mode cannot reuse server-side prepared
        # statements across pooled backends.
        url = url.update_query_dict({"prepared_statement_cache_size": "0"})
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid.uuid4()}__"
    return create_async_engine(
        url,
        connect_args=connect_args,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
