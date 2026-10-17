"""CLI entry: python -m evaluation.run"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import uuid
from pathlib import Path

from evaluation.datasets.load import load_dataset
from evaluation.metrics.aggregate import aggregate_report
from evaluation.reports.writer import write_report
from evaluation.runners.offline import OfflineRunConfig, run_case_offline
from evaluation.schemas.models import ExecutionStatus, RunMetadata, RunReport


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def _run_offline(
    cases, *, mode: str, limit: int | None, dataset_path: str
) -> RunReport:
    if limit:
        cases = cases[:limit]
    results = []
    for case in cases:
        if case.offline_only and mode == "baseline-live":
            continue
        result = await run_case_offline(case, mode=mode.split("-")[-1] if "-" in mode else mode)
        results.append(result)
    meta = RunMetadata(
        dataset_path=dataset_path,
        mode=mode,
        live_llm=False,
        git_commit=_git_commit(),
        case_count=len(results),
        success_count=sum(1 for r in results if r.execution_status == ExecutionStatus.SUCCESS),
        failed_count=sum(1 for r in results if r.execution_status == ExecutionStatus.FAILED),
    )
    report = RunReport(metadata=meta, results=results)
    return aggregate_report(report)


async def _run_live(cases, *, mode: str, limit: int | None, dataset_path: str) -> RunReport:
    from app.core.config import get_settings
    from app.db.session import create_engine, create_session_factory
    from app.services.context_builder import ContextBuilder
    from app.services.embedding_service import OpenAIEmbeddingService
    from app.services.generation_service import OpenAIGenerationService
    from app.services.retrieval_service import RetrievalService
    from evaluation.runners.baseline import run_case_baseline
    from openai import AsyncOpenAI

    settings = get_settings()
    if limit:
        cases = cases[:limit]
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    embedder = OpenAIEmbeddingService.from_settings(settings)
    llm = AsyncOpenAI(api_key=settings.require_openai(), base_url=settings.embedding_api_base())
    generator = OpenAIGenerationService.from_settings(settings, llm)
    retrieval = RetrievalService(
        embedder,
        top_k=settings.retrieval_top_k,
        max_candidates=settings.retrieval_max_candidates,
        embedding_model=settings.openai_embedding_model,
    )
    context_builder = ContextBuilder(
        max_tokens=settings.max_context_tokens,
        embedding_model=settings.openai_embedding_model,
    )
    owner = settings.dev_owner_id
    results = []
    try:
        for case in cases:
            if mode in ("baseline", "both"):
                results.append(
                    await run_case_baseline(
                        case,
                        session_factory=session_factory,
                        retrieval=retrieval,
                        generator=generator,
                        context_builder=context_builder,
                        owner_id=owner,
                    )
                )
    finally:
        await embedder.aclose()
        await llm.close()
        await engine.dispose()
    meta = RunMetadata(
        dataset_path=dataset_path,
        mode=f"live-{mode}",
        live_llm=True,
        git_commit=_git_commit(),
        case_count=len(results),
        success_count=sum(1 for r in results if r.execution_status == ExecutionStatus.SUCCESS),
        failed_count=sum(1 for r in results if r.execution_status == ExecutionStatus.FAILED),
        model_config_snapshot={"chat_model": settings.openai_chat_model},
    )
    return aggregate_report(RunReport(metadata=meta, results=results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AI Twin evaluation suite")
    parser.add_argument("--dataset", default="evaluation/datasets/core.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--live-llm", action="store_true", help="Use live LLM + DB (opt-in)")
    parser.add_argument("--mode", choices=["offline", "baseline", "agentic", "both"], default="offline")
    parser.add_argument("--output", default="reports")
    parsed = parser.parse_args(argv)
    dataset_path = Path(parsed.dataset)
    cases = load_dataset(dataset_path)
    ds = str(dataset_path)
    if parsed.live_llm:
        report = asyncio.run(_run_live(cases, mode=parsed.mode, limit=parsed.limit, dataset_path=ds))
    else:
        run_mode = "agentic" if parsed.mode == "agentic" else "baseline"
        if parsed.mode == "offline":
            run_mode = "baseline"
        report = asyncio.run(
            _run_offline(cases, mode=run_mode, limit=parsed.limit, dataset_path=ds)
        )
    json_path, md_path = write_report(report, Path(parsed.output))
    print(f"Wrote {json_path} and {md_path}")
    print("Aggregates:", report.aggregates)
    failed = report.metadata.failed_count
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
