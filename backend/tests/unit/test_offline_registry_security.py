import uuid

from evaluation.fixtures.offline_registry import build_default_registry


def test_private_chunk_not_retrieved_for_owner() -> None:
    reg = build_default_registry()
    hits = reg.retrieve("employee SSN", top_k=8, owner_scope=reg.owner_id)
    assert all(h.chunk_id not in reg.private_chunk_ids for h in hits)


def test_cross_owner_retrieval_empty() -> None:
    reg = build_default_registry()
    other = uuid.UUID("00000000-0000-0000-0000-000000000099")
    assert reg.retrieve("remote work", top_k=8, owner_scope=other) == []
