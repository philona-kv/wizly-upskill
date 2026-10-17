"""Aggregate evaluation results into report summaries."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from evaluation.schemas.models import CategoryAggregate, EvalResult, RunReport


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def _mean_bool(results: list[EvalResult], name: str) -> float | None:
    vals = []
    for result in results:
        value = getattr(result.scores, name, None)
        if value is not None:
            vals.append(1.0 if value else 0.0)
    return mean(vals) if vals else None


def _mean_field(results: list[EvalResult], name: str) -> float | None:
    vals = []
    for result in results:
        value = getattr(result.scores, name, None)
        if value is not None:
            vals.append(float(value))
    return mean(vals) if vals else None


def aggregate_report(report: RunReport) -> RunReport:
    results = report.results
    latencies = [float(r.latency_ms_total) for r in results if r.latency_ms_total > 0]
    report.aggregates = {
        "recall_at_k": _mean_field(results, "recall_at_k"),
        "precision_at_k": _mean_field(results, "precision_at_k"),
        "mrr": _mean_field(results, "mrr"),
        "key_fact_recall": _mean_field(results, "key_fact_recall"),
        "faithfulness_offline": _mean_field(results, "faithfulness_offline"),
        "abstention_accuracy": _mean_bool(results, "abstention_correct"),
    }
    report.latency_ms = {
        "median": percentile(latencies, 50),
        "p90": percentile(latencies, 90),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99) if len(latencies) >= 20 else None,
        "sample_count": float(len(latencies)),
    }
    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        cat = str(result.metadata.get("category", "unknown"))
        by_cat[cat].append(result)
    report.aggregates_by_category = [
        CategoryAggregate(
            category=cat,
            count=len(items),
            metrics={
                "recall_at_k": _mean_field(items, "recall_at_k"),
                "key_fact_recall": _mean_field(items, "key_fact_recall"),
            },
        )
        for cat, items in sorted(by_cat.items())
    ]
    return report
