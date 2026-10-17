"""Compare two evaluation JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.schemas.models import CompareReport, RunReport


def compare_reports(baseline: RunReport, candidate: RunReport) -> CompareReport:
    deltas: dict[str, float | None] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    keys = set(baseline.aggregates) | set(candidate.aggregates)
    for key in sorted(keys):
        b = baseline.aggregates.get(key)
        c = candidate.aggregates.get(key)
        if b is None or c is None:
            deltas[key] = None
            continue
        delta = c - b
        deltas[key] = delta
        if delta < -0.05:
            regressions.append(f"{key}: {b:.3f} -> {c:.3f}")
        elif delta > 0.05:
            improvements.append(f"{key}: {b:.3f} -> {c:.3f}")
    return CompareReport(
        baseline_path="",
        candidate_path="",
        metric_deltas=deltas,
        regressions=regressions,
        improvements=improvements,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare evaluation reports")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    baseline = RunReport.model_validate(json.loads(Path(args.baseline).read_text(encoding="utf-8")))
    candidate = RunReport.model_validate(json.loads(Path(args.candidate).read_text(encoding="utf-8")))
    report = compare_reports(baseline, candidate)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 1 if report.regressions else 0


if __name__ == "__main__":
    sys.exit(main())
