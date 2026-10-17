import uuid

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.db.models import Document, DocumentChunk
from app.services.retrieval_service import retrieval_filters


def test_retrieval_filters_are_owner_global_ready_and_matching_model() -> None:
    owner = uuid.UUID("00000000-0000-0000-0000-00000000000a")
    stmt = (
        select(DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(*retrieval_filters(owner, "text-embedding-3-small"))
    )
    sql = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    ).lower()
    assert str(owner) in sql
    assert "global" in sql
    assert "ready" in sql
    assert "text-embedding-3-small" in sql
    assert "private" not in sql
    assert "owner_id" in sql
    assert "embedding_model" in sql
