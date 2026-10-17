"""Offline-first evaluation framework for the AI Twin (Subtask 5)."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if _BACKEND.is_dir() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

__all__ = ["__version__"]

__version__ = "0.1.0"
