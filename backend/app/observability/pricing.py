"""Configurable model pricing for cost estimates."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[3] / "evaluation" / "pricing" / "models.yaml"


@lru_cache
def _load_registry(path: str | None = None) -> dict:
    pricing_path = Path(path) if path else DEFAULT_PRICING_PATH
    if not pricing_path.is_file():
        return {"models": {}}
    data = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
    return data


def estimate_cost_usd(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int | None,
    *,
    pricing_path: str | None = None,
) -> float | None:
    registry = _load_registry(pricing_path)
    models = registry.get("models") or {}
    entry = models.get(model)
    if not entry or input_tokens is None:
        return None
    in_price = float(entry.get("input_per_million", 0))
    out_price = float(entry.get("output_per_million", 0))
    cached_price = float(entry.get("cached_input_per_million", in_price))
    out_tokens = output_tokens or 0
    cached = cached_tokens or 0
    regular_in = max(0, input_tokens - cached)
    cost = (regular_in / 1_000_000) * in_price + (cached / 1_000_000) * cached_price
    cost += (out_tokens / 1_000_000) * out_price
    return round(cost, 6)
