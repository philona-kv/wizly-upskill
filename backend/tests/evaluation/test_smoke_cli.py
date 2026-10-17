import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_offline_eval_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation",
            "--dataset",
            "evaluation/datasets/core.jsonl",
            "--limit",
            "5",
            "--offline",
            "--output",
            str(ROOT / "reports" / "pytest-smoke"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
