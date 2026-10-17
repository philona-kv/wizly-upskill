"""Optional Postgres LangGraph checkpointer (setup at app startup)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.core.config import Settings

logger = logging.getLogger(__name__)

# asyncpg query keys that psycopg/libpq conn strings do not accept.
_PSYCHOPG_DROP_QUERY_KEYS = frozenset({"ssl", "prepared_statement_cache_size"})


def libpq_database_url(settings: Settings) -> str:
    """Connection string for psycopg (LangGraph checkpoint saver).

    ``Settings.database_url`` is normalised for SQLAlchemy asyncpg (``ssl=``).
    psycopg expects libpq-style ``sslmode=`` instead.
    """
    raw = settings.langgraph_checkpoint_database_url.strip() or (settings.database_url or "")
    url = raw.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(url)
    query: list[tuple[str, str]] = []
    if parsed.query:
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            lowered = key.lower()
            if lowered == "ssl":
                query.append(("sslmode", value))
                continue
            if lowered in _PSYCHOPG_DROP_QUERY_KEYS:
                continue
            query.append((key, value))
    existing_keys = {k.lower() for k, _ in query}
    if "prepare_threshold" not in existing_keys:
        query.append(("prepare_threshold", "0"))
    return urlunparse(parsed._replace(query=urlencode(query)))


async def _setup_checkpointer(checkpointer: Any) -> None:
    """Run LangGraph checkpoint migrations; recover from pooler prepared-statement clashes."""
    from psycopg.errors import DuplicatePreparedStatement

    try:
        await checkpointer.setup()
    except DuplicatePreparedStatement:
        logger.warning(
            "checkpoint_setup_deallocate_retry",
            extra={"reason": "duplicate_prepared_statement"},
        )
        conn = checkpointer.conn
        async with conn.cursor() as cur:
            await cur.execute("DEALLOCATE ALL")
        await checkpointer.setup()


@asynccontextmanager
async def open_postgres_checkpointer(settings: Settings) -> AsyncIterator[Any]:
    """Yield an ``AsyncPostgresSaver`` after ``setup()``, or ``None`` when disabled."""
    if not settings.langgraph_checkpointing_enabled:
        yield None
        return
    if not settings.database_url:
        logger.warning("checkpoint_skipped", extra={"reason": "no_database_url"})
        yield None
        return
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
    except ImportError:
        logger.warning(
            "checkpoint_skipped",
            extra={"reason": "langgraph-checkpoint-postgres not installed"},
        )
        yield None
        return

    uri = libpq_database_url(settings)
    try:
        async with await AsyncConnection.connect(
            uri,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await _setup_checkpointer(checkpointer)
            yield checkpointer
    except Exception as exc:
        logger.error(
            "checkpoint_setup_failed",
            extra={"exc_type": type(exc).__name__},
        )
        if settings.app_env.value == "production":
            raise
        logger.warning(
            "checkpoint_disabled",
            extra={
                "hint": "Use a direct Postgres URL (not transaction pooler) for "
                "LANGGRAPH_CHECKPOINT_DATABASE_URL, or set LANGGRAPH_CHECKPOINTING_ENABLED=false.",
            },
        )
        yield None
