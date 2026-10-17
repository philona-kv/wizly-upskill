"""Seed synthetic fixtures into TEST_DATABASE_URL (optional DB eval mode)."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Document, DocumentChunk, DocumentStatus, DocumentVisibility
from evaluation.fixtures.offline_registry import build_default_registry


async def seed_offline_documents(
    session_factory: async_sessionmaker[AsyncSession],
    owner_id: uuid.UUID,
) -> dict[str, str]:
    """Insert global ready documents from the offline registry. Returns fixture_doc_id -> document_id."""
    registry = build_default_registry()
    mapping: dict[str, str] = {}
    async with session_factory() as session, session.begin():
        for chunk in registry.chunks:
            if chunk.chunk_id in registry.private_chunk_ids:
                continue
            fixture_key = chunk.original_filename.replace(".txt", "")
            doc = Document(
                id=uuid.UUID(chunk.document_id),
                owner_id=owner_id,
                title=chunk.document_title,
                original_filename=chunk.original_filename,
                storage_path=f"eval/{chunk.document_id}.txt",
                mime_type="text/plain",
                file_size_bytes=len(chunk.content.encode("utf-8")),
                file_hash=f"eval-{chunk.document_id}",
                visibility=DocumentVisibility.GLOBAL.value,
                status=DocumentStatus.READY.value,
                chunk_count=1,
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536,
            )
            session.add(doc)
            session.add(
                DocumentChunk(
                    id=uuid.UUID(chunk.chunk_id),
                    document_id=doc.id,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_index=0,
                    token_count=50,
                    embedding=[0.01] + [0.0] * 1535,
                )
            )
            mapping[fixture_key] = str(doc.id)
    return mapping
