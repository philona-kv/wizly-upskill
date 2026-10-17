"""Load and validate evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evaluation.schemas.models import EvalCase


def load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
                cases.append(EvalCase.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(f"line {line_no}: {exc}")
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases
