from app.core.config import Settings
from app.workflows.checkpointer import libpq_database_url


def test_libpq_url_converts_asyncpg_ssl_to_sslmode() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://user:pass@host.example.com:5432/db?ssl=require"
    )
    url = libpq_database_url(settings)
    assert url.startswith("postgresql://user:pass@host.example.com:5432/db?")
    assert "sslmode=require" in url
    assert "prepare_threshold=0" in url
