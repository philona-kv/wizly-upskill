import pytest
from pydantic import ValidationError

from app.core.config import ConfigurationError, Settings, to_async_database_url

_SETTINGS_ENV_KEYS = (
    "OPENAI_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_API_BASE",
    "DATABASE_URL",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgresql://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgres://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        ("postgresql+psycopg2://u@h/db", "postgresql+asyncpg://u@h/db"),
        ("postgresql+asyncpg://u@h/db", "postgresql+asyncpg://u@h/db"),
        ("postgresql://u@h/db?sslmode=require", "postgresql+asyncpg://u@h/db?ssl=require"),
        ("", ""),
    ],
)
def test_async_url_conversion(raw: str, expected: str) -> None:
    assert to_async_database_url(raw) == expected


def test_rejects_non_postgres_urls() -> None:
    with pytest.raises(ValueError):
        to_async_database_url("mysql://u@h/db")


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        _settings(chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_dev_auth_forbidden_in_production() -> None:
    with pytest.raises(ValidationError):
        _settings(app_env="production", auth_mode="development")


def test_require_helpers_validate_lazily() -> None:
    settings = _settings(database_url="", openai_api_key="", supabase_url="")
    with pytest.raises(ConfigurationError):
        settings.require_database()
    with pytest.raises(ConfigurationError):
        settings.require_openai()
    with pytest.raises(ConfigurationError):
        settings.require_supabase()


def test_secrets_are_not_rendered() -> None:
    settings = _settings(openai_api_key="sk-very-secret", supabase_service_role_key="srk-secret")
    rendered = repr(settings) + str(settings.model_dump())
    assert "sk-very-secret" not in rendered
    assert "srk-secret" not in rendered
    assert settings.require_openai() == "sk-very-secret"


def test_max_upload_bytes() -> None:
    assert _settings(max_upload_size_mb=2).max_upload_size_bytes == 2 * 1024 * 1024


def test_litellm_base_url_normalized() -> None:
    settings = _settings(litellm_api_base="http://localhost:4000")
    assert settings.embedding_api_base() == "http://localhost:4000/v1"
    assert settings.uses_litellm is True


def test_litellm_accepts_litellm_or_openai_key() -> None:
    litellm_only = _settings(
        litellm_api_base="http://localhost:4000",
        litellm_api_key="litellm-master",
        openai_api_key="",
    )
    assert litellm_only.require_openai() == "litellm-master"
    fallback = _settings(
        litellm_api_base="http://localhost:4000",
        openai_api_key="sk-openai",
    )
    assert fallback.require_openai() == "sk-openai"


def test_retrieval_candidates_must_meet_top_k() -> None:
    with pytest.raises(ValidationError):
        _settings(retrieval_top_k=10, retrieval_max_candidates=5)


def test_litellm_requires_some_key() -> None:
    with pytest.raises(ConfigurationError):
        _settings(litellm_api_base="http://localhost:4000", openai_api_key="").require_openai()
