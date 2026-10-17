"""Write evaluation reports to disk."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from evaluation.schemas.models import RunReport


def write_report(report: RunReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.metadata.timestamp.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"run-{stamp}.json"
    md_path = output_dir / f"run-{stamp}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    return json_path, md_path


def _markdown_summary(report: RunReport) -> str:
    meta = report.metadata
    lines = [
        "# AI Twin evaluation report",
        "",
        f"- Run ID: `{meta.run_id}`",
        f"- Timestamp: {meta.timestamp.astimezone(UTC).isoformat()}",
        f"- Mode: {meta.mode}",
        f"- Dataset: {meta.dataset_path}",
        f"- Cases: {meta.case_count} (success {meta.success_count}, failed {meta.failed_count})",
        "",
        "## Aggregate metrics",
    ]
    for key, value in sorted(report.aggregates.items()):
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Latency (ms)"])
    for key, value in sorted(report.latency_ms.items()):
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## By category"])
    for item in report.aggregates_by_category:
        lines.append(f"- **{item.category}** (n={item.count}): {item.metrics}")
    lines.append("")
    lines.append("_LLM-judge and cost figures are estimates when labeled as such._")
    return "\n".join(lines)
