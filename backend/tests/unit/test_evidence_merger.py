import uuid

from app.services.evidence_merger import assign_citation_ids, merge_evidence_lists
from app.workflows.schemas import CompactMessage, EvidenceItem, EvidenceSourceType


def test_merge_dedupes_chunks() -> None:
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    item = EvidenceItem(
        evidence_id=str(chunk_id),
        source_type=EvidenceSourceType.DOCUMENT,
        content="a",
        document_id=doc_id,
        chunk_id=chunk_id,
    )
    merged = merge_evidence_lists([item, item.model_copy()], [], max_items=10)
    assert len(merged) == 1


def test_assign_citation_ids() -> None:
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    item = EvidenceItem(
        evidence_id=str(chunk_id),
        source_type=EvidenceSourceType.DOCUMENT,
        content="policy text",
        document_id=doc_id,
        chunk_id=chunk_id,
    )
    cited = assign_citation_ids([item])
    assert cited[0].citation_id == "S1"


def test_messages_to_evidence_bounded() -> None:
    from app.services.evidence_merger import messages_to_evidence

    cid = uuid.uuid4()
    msgs = [
        CompactMessage(id=uuid.uuid4(), role="user", content="hello"),
        CompactMessage(id=uuid.uuid4(), role="assistant", content="hi"),
    ]
    items = messages_to_evidence(msgs, conversation_id=cid, max_messages=1)
    assert len(items) == 1
