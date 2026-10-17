"""Deterministic answer and citation validation."""

from __future__ import annotations

import re
from typing import Protocol

from app.services.citation_validator import validate_answer
from app.services.context_builder import EvidenceItem as DocEvidence
from app.workflows.schemas import AnswerValidation, EvidenceItem

_CITE_RE = re.compile(r"\[(S\d+)\]")


class AnswerValidator(Protocol):
    def validate(
        self,
        *,
        answer: str,
        document_evidence: list[EvidenceItem],
        doc_context_evidence: list[DocEvidence],
    ) -> AnswerValidation: ...


class DeterministicAnswerValidator:
    def validate(
        self,
        *,
        answer: str,
        document_evidence: list[EvidenceItem],
        doc_context_evidence: list[DocEvidence],
    ) -> AnswerValidation:
        if not answer.strip():
            return AnswerValidation(
                valid=False,
                correction_required=True,
                explanation="Empty answer.",
            )
        allowed_ids = {item.citation_id for item in document_evidence if item.citation_id}
        mentioned = list(dict.fromkeys(_CITE_RE.findall(answer)))
        invalid = [cid for cid in mentioned if cid not in allowed_ids]
        if invalid:
            return AnswerValidation(
                valid=False,
                invalid_citations=invalid,
                correction_required=True,
                explanation="Unknown citation identifiers.",
            )
        validated = validate_answer(answer, doc_context_evidence)
        if not validated.content.strip():
            return AnswerValidation(
                valid=False,
                correction_required=True,
                explanation="Answer invalid after citation cleanup.",
            )
        doc_ids = {str(item.document_id) for item in document_evidence if item.document_id}
        for cite in validated.citations:
            if str(cite.document_id) not in doc_ids:
                return AnswerValidation(
                    valid=False,
                    invalid_citations=[cite.citation_id],
                    correction_required=True,
                    explanation="Citation document not in evidence set.",
                )
        return AnswerValidation(valid=True, explanation="Deterministic checks passed.")
