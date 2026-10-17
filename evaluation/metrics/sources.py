"""Source-selection accuracy for agentic routing."""

from __future__ import annotations


def source_selection_match(required: list[str], selected: list[str]) -> bool | None:
    if not required:
        return None
    required_set = set(required)
    selected_set = set(selected)
    return required_set.issubset(selected_set)
