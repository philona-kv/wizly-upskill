"""Retrieval ranking metrics (binary relevance unless grades provided)."""

from __future__ import annotations

import math


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_grades: dict[str, float], retrieved: list[str], k: int) -> float | None:
    if not relevant_grades:
        return None
    top = retrieved[:k]

    def dcg(items: list[str]) -> float:
        total = 0.0
        for i, item in enumerate(items, start=1):
            grade = relevant_grades.get(item, 0.0)
            if grade > 0:
                total += (2**grade - 1) / math.log2(i + 1)
        return total

    ideal = sorted(relevant_grades.items(), key=lambda x: x[1], reverse=True)
    ideal_ids = [item for item, grade in ideal if grade > 0][:k]
    idcg = dcg(ideal_ids)
    if idcg == 0:
        return 0.0
    return dcg(top) / idcg
