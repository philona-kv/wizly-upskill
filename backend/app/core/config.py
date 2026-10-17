"""Centralised application settings (single source: environment variables / backend/.env)."""

from __future__ import annotations

import uuid
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class AuthMode(StrEnum):
    DEVELOPMENT = "development"
    DISABLED = "disabled"


class ConfigurationError(RuntimeError):
    """A dependent service was initialised without the configuration it requires."""


def to_async_database_url(url: str) -> str:
    """Convert a libpq-style URL into the SQLAlchemy asyncpg driver form.

    Supabase and most providers hand out ``postgres://`` or ``postgresql://`` URLs,
    which SQLAlchemy maps to the synchronous psycopg2 driver. The application uses
    asyncpg everywhere (including Alembic), so the scheme is rewritten explicitly.
    ``sslmode`` is a libpq parameter that asyncpg does not accept; it is renamed to ``ssl``.
    """
    url = url.strip()
    if not url:
        return url
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError("DATABASE_URL must be a URL such as postgresql://user:pass@host/db")
    scheme = scheme.lower()
    if scheme in {"postgres", "postgresql", "postgresql+psycopg2", "postgresql+psycopg"}:
        scheme = "postgresql+asyncpg"
    elif scheme != "postgresql+asyncpg":
        raise ValueError(f"Unsupported database scheme '{scheme}'; PostgreSQL is required")
    rest = rest.replace("sslmode=", "ssl=")
    return f"{scheme}://{rest}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 - bind address for local dev servers
    api_port: int = 8000

    database_url: str = ""
    database_use_pgbouncer: bool = False
    database_pool_size: int = 5
    database_echo: bool = False

    openai_api_key: SecretStr | None = None
    # LiteLLM proxy (OpenAI-compatible). When LITELLM_API_BASE is set, embeddings use it
    # instead of https://api.openai.com. See backend/.env.example.
    litellm_api_base: str = ""
    litellm_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = Field(default=1536, gt=0)
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_max_retries: int = Field(default=4, ge=0, le=10)
    embedding_batch_size: int = Field(default=96, gt=0, le=2048)
    embedding_max_batch_tokens: int = Field(default=250_000, gt=0)

    openai_chat_model: str = "gpt-6.1-sol"
    retrieval_top_k: int = Field(default=8, gt=0, le=50)
    retrieval_max_candidates: int = Field(default=30, gt=0, le=200)
    max_context_tokens: int = Field(default=10_000, gt=0)
    max_chat_history_messages: int = Field(default=10, gt=0, le=50)
    chat_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    langgraph_max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    langgraph_max_answer_repairs: int = Field(default=1, ge=0, le=3)
    langgraph_enable_debug_metadata: bool = False
    langgraph_checkpointing_enabled: bool = False
    # Optional direct Postgres URL for LangGraph checkpoints (recommended for Supabase:
    # use session/direct port 5432, not the transaction pooler on 6543).
    langgraph_checkpoint_database_url: str = ""
    langgraph_workflow_timeout_seconds: float = Field(default=120.0, gt=0)
    langgraph_max_evidence_items: int = Field(default=24, gt=0, le=100)
    langgraph_graph_version: str = "twin-v4"
    langgraph_recovery_enabled: bool = True
    langgraph_max_recovery_attempts: int = Field(default=2, ge=0, le=5)
    langgraph_retry_backoff_seconds: float = Field(default=1.0, ge=0)

    memory_enabled: bool = True
    memory_extraction_enabled: bool = True
    memory_extraction_interval_messages: int = Field(default=6, ge=2, le=100)
    memory_retrieval_top_k: int = Field(default=8, gt=0, le=50)
    memory_max_records_per_user: int = Field(default=500, gt=0)
    memory_min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_embedding_model: str = ""
    memory_context_token_budget: int = Field(default=2000, gt=0)

    conversation_summary_enabled: bool = True
    conversation_summary_trigger_messages: int = Field(default=12, ge=4, le=200)
    conversation_summary_max_tokens: int = Field(default=800, gt=0)
    conversation_summary_incremental: bool = True

    supabase_url: str = ""
    supabase_service_role_key: SecretStr | None = None
    supabase_storage_bucket: str = "documents"
    storage_timeout_seconds: float = Field(default=60.0, gt=0)

    max_upload_size_mb: int = Field(default=25, gt=0)
    chunk_size_tokens: int = Field(default=650, gt=0)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    ingestion_stale_after_seconds: int = Field(default=900, gt=0)

    auth_mode: AuthMode = AuthMode.DEVELOPMENT
    dev_owner_id: uuid.UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    @field_validator("database_url")
    @classmethod
    def _convert_database_url(cls, value: str) -> str:
        return to_async_database_url(value)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("supabase_url")
    @classmethod
    def _strip_supabase_url(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("litellm_api_base")
    @classmethod
    def _strip_litellm_base(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.retrieval_max_candidates < self.retrieval_top_k:
            raise ValueError("RETRIEVAL_MAX_CANDIDATES must be at least RETRIEVAL_TOP_K")
        if self.app_env == AppEnv.PRODUCTION and self.auth_mode == AuthMode.DEVELOPMENT:
            raise ValueError("AUTH_MODE=development is not allowed when APP_ENV=production")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # Validation is deferred to the point of use so that, e.g., the health
    # endpoint and unit tests work without every credential being present.
    def require_database(self) -> str:
        if not self.database_url:
            raise ConfigurationError("DATABASE_URL is not configured")
        return self.database_url

    @property
    def uses_litellm(self) -> bool:
        return bool(self.litellm_api_base)

    def embedding_api_base(self) -> str | None:
        """OpenAI SDK ``base_url`` (…/v1) when routing through LiteLLM, else ``None``."""
        if not self.litellm_api_base:
            return None
        base = self.litellm_api_base.rstrip("/")
        for suffix in ("/chat/completions", "/embeddings"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        return base if base.endswith("/v1") else f"{base}/v1"

    def memory_embedding_model_name(self) -> str:
        return self.memory_embedding_model.strip() or self.openai_embedding_model

    def require_openai(self) -> str:
        """API key for embeddings (OpenAI direct or LiteLLM proxy)."""
        if self.uses_litellm:
            key = self.litellm_api_key or self.openai_api_key
            if key is None or not key.get_secret_value():
                raise ConfigurationError(
                    "LITELLM_API_KEY or OPENAI_API_KEY is required when LITELLM_API_BASE is set"
                )
            return key.get_secret_value()
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            raise ConfigurationError("OPENAI_API_KEY is not configured")
        return self.openai_api_key.get_secret_value()

    def require_supabase(self) -> tuple[str, str]:
        key = self.supabase_service_role_key
        if not self.supabase_url or key is None or not key.get_secret_value():
            raise ConfigurationError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured"
            )
        return self.supabase_url, key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
