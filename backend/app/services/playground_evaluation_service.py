"""Run evaluation datasets through the Playground query path."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.schemas.playground import EvaluationRunRequest, EvaluationRunResponse, PlaygroundQueryRequest
from app.services.playground_document_access import PlaygroundDocumentAccessService
from app.services.playground_query_service import PlaygroundQueryService


class PlaygroundEvaluationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        query_service: PlaygroundQueryService,
        *,
        repo_root: Path,
    ) -> None:
        self._session_factory = session_factory
        self._query = query_service
        self._repo_root = repo_root
        self._access = PlaygroundDocumentAccessService()

    async def run(self, owner_id: uuid.UUID, body: EvaluationRunRequest) -> EvaluationRunResponse:
        import sys

        root = str(self._repo_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from evaluation.datasets.load import load_dataset

        dataset_path = self._repo_root / body.dataset_path
        if not dataset_path.is_file():
            from app.core.exceptions import ValidationFailedError

            raise ValidationFailedError(f"Dataset not found: {body.dataset_path}")
        cases = load_dataset(dataset_path)
        if body.limit:
            cases = cases[: body.limit]

        async with self._session_factory() as session:
            await self._access.validate_scope(
                session, owner_id, body.scope, body.document_ids
            )

        run_id = uuid.uuid4()
        results: list[dict] = []
        latencies: list[int] = []
        for case in cases:
            started_ms = 0
            import time

            t0 = time.perf_counter()
            try:
                resp = await self._query.query(
                    owner_id,
                    PlaygroundQueryRequest(
                        question=case.question,
                        scope=body.scope,
                        document_ids=body.document_ids,
                        prompt_version_id=body.prompt_version_id,
                    ),
                )
                latency = int((time.perf_counter() - t0) * 1000)
                latencies.append(latency)
                row = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "answer": resp.answer,
                    "status": resp.status,
                    "latency_ms": latency,
                    "citation_count": len(resp.citations),
                }
                if case.expected_key_facts:
                    from evaluation.metrics.answers import key_fact_recall

                    score = key_fact_recall(resp.answer, case.expected_key_facts)
                    if score is not None:
                        row["key_fact_recall"] = score
                results.append(row)
            except Exception as exc:
                results.append(
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "status": "failed",
                        "error": type(exc).__name__,
                    }
                )

        aggregate: dict = {"case_count": len(results)}
        if latencies:
            aggregate["mean_latency_ms"] = sum(latencies) / len(latencies)
        recalls = [r["key_fact_recall"] for r in results if "key_fact_recall" in r]
        if recalls:
            aggregate["mean_key_fact_recall"] = sum(recalls) / len(recalls)

        return EvaluationRunResponse(
            run_id=run_id,
            case_count=len(results),
            aggregate_metrics=aggregate,
            results=results,
        )
