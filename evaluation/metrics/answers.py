"""Deterministic answer-quality metrics."""

from __future__ import annotations

import re

from evaluation.schemas.models import EvalCase, MetricScores

_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def key_fact_recall(answer: str, expected_facts: list[str]) -> float | None:
    if not expected_facts:
        return None
    text = answer.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in text)
    return hits / len(expected_facts)


def offline_faithfulness(answer: str, evidence_texts: list[str]) -> float | None:
    tokens = _NUMERIC_RE.findall(answer)
    if not tokens:
        return None
    corpus = " ".join(evidence_texts).lower()
    supported = sum(1 for t in tokens if t in corpus)
    return supported / len(tokens)


def abstention_correct(should_abstain: bool, actual_abstention: bool) -> bool:
    return should_abstain == actual_abstention


def score_answer_metrics(
    case: EvalCase,
    *,
    answer: str,
    actual_abstention: bool,
    citation_valid: bool | None,
    evidence_texts: list[str],
) -> MetricScores:
    return MetricScores(
        key_fact_recall=key_fact_recall(answer, case.expected_key_facts),
        abstention_correct=abstention_correct(case.should_abstain, actual_abstention),
        citation_valid=citation_valid,
        faithfulness_offline=offline_faithfulness(answer, evidence_texts),
    )
