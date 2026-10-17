"""Keep only citations that map to this turn's retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.conversation import Citation
from app.services.context_builder import EvidenceItem

_CITE_RE = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True, slots=True)
class ValidatedAnswer:
    content: str
    citations: list[Citation]


def _citation_from_evidence(item: EvidenceItem) -> Citation:
    chunk = item.chunk
    return Citation(
        citation_id=item.citation_id,
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        original_filename=chunk.original_filename,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_id=chunk.chunk_id,
        excerpt=item.excerpt,
    )


def validate_answer(content: str, evidence: list[EvidenceItem]) -> ValidatedAnswer:
    allowed = {item.citation_id: item for item in evidence}
    mentioned = list(dict.fromkeys(_CITE_RE.findall(content)))
    used: list[EvidenceItem] = []
    cleaned = content
    for cid in mentioned:
        if cid in allowed:
            used.append(allowed[cid])
        else:
            cleaned = cleaned.replace(f"[{cid}]", "")
    cleaned = re.sub(r" +", " ", cleaned).strip()
    if not cleaned:
        cleaned = content.strip()
    return ValidatedAnswer(
        content=cleaned, citations=[_citation_from_evidence(item) for item in used]
    )
