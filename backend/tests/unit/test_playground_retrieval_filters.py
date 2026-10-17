import uuid

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.db.models import Document, DocumentChunk
from app.repositories.document_repository import playground_retrieval_filter
from app.schemas.playground import PlaygroundRetrievalScope
from app.services.retrieval_service import RetrievalContext, retrieval_filters


def test_playground_private_filter_sql() -> None:
    owner = uuid.UUID("00000000-0000-0000-0000-00000000000a")
    ctx = RetrievalContext.playground(PlaygroundRetrievalScope.PRIVATE_ONLY)
    stmt = (
        select(DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(*retrieval_filters(owner, "text-embedding-3-small", context=ctx))
    )
    sql = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    ).lower()
    assert "private" in sql
    assert "ready" in sql
    assert str(owner) in sql


def test_selected_scope_requires_document_ids_in_filter() -> None:
    doc_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    filt = playground_retrieval_filter(
        PlaygroundRetrievalScope.SELECTED, document_ids=[doc_id]
    )
    sql = str(filt).lower()
    assert "documents.id in" in sql
    assert "status" in sql
