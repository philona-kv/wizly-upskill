import uuid

from app.services.retrieval_service import RetrievedChunk, reciprocal_rank_fusion


def _chunk(label: str, rank_distance: float) -> RetrievedChunk:
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, label)
    did = uuid.uuid5(uuid.NAMESPACE_DNS, label + "-doc")
    return RetrievedChunk(
        chunk_id=cid,
        document_id=did,
        document_title="T",
        original_filename="f.pdf",
        content=label,
        page_start=1,
        page_end=1,
        chunk_index=0,
        distance=rank_distance,
    )


def test_rrf_boosts_overlap_between_lists() -> None:
    a, b, c = _chunk("a", 0.1), _chunk("b", 0.2), _chunk("c", 0.3)
    semantic = [a, c, b]
    fulltext = [b, c]
    fused = reciprocal_rank_fusion([semantic, fulltext], top_k=3)
    assert [x.content for x in fused] == ["b", "c", "a"]


def test_rrf_respects_top_k() -> None:
    chunks = [_chunk(f"x{i}", float(i)) for i in range(5)]
    fused = reciprocal_rank_fusion([chunks, list(reversed(chunks))], top_k=3)
    assert len(fused) == 3
