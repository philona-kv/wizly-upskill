from __future__ import annotations

import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError, ValidationFailedError
from app.db.models.playground import PlaygroundExperiment, PlaygroundExperimentRun
from app.repositories.playground_repository import PlaygroundExperimentRepository
from app.schemas.playground import ExperimentRunRequest
from app.services.playground_document_access import PlaygroundDocumentAccessService
from app.services.playground_query_service import PlaygroundQueryService
from app.services.prompt_template_service import PromptTemplateService


class PlaygroundExperimentService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        query_service: PlaygroundQueryService,
        prompt_service: PromptTemplateService,
    ) -> None:
        self._session_factory = session_factory
        self._query = query_service
        self._prompts = prompt_service
        self._access = PlaygroundDocumentAccessService()

    async def create(self, owner_id: uuid.UUID, name: str) -> PlaygroundExperiment:
        experiment = PlaygroundExperiment(id=uuid.uuid4(), owner_id=owner_id, name=name)
        async with self._session_factory() as session, session.begin():
            PlaygroundExperimentRepository(session).add(experiment)
        return experiment

    async def list_experiments(
        self, owner_id: uuid.UUID, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[PlaygroundExperiment], int]:
        async with self._session_factory() as session:
            return await PlaygroundExperimentRepository(session).list_for_owner(
                owner_id, limit=limit, offset=offset
            )

    async def get(self, owner_id: uuid.UUID, experiment_id: uuid.UUID) -> PlaygroundExperiment:
        async with self._session_factory() as session:
            exp = await PlaygroundExperimentRepository(session).get_for_owner(
                experiment_id, owner_id
            )
            if exp is None:
                raise NotFoundError("Experiment not found.")
            return exp

    async def run_experiment(
        self, owner_id: uuid.UUID, experiment_id: uuid.UUID, body: ExperimentRunRequest
    ) -> list[PlaygroundExperimentRun]:
        await self.get(owner_id, experiment_id)
        async with self._session_factory() as session:
            await self._access.validate_scope(
                session, owner_id, body.scope, body.document_ids
            )
        runs: list[PlaygroundExperimentRun] = []
        for version_id in body.prompt_version_ids:
            version = await self._prompts.get_version_for_owner(owner_id, version_id)
            started = time.perf_counter()
            from app.schemas.playground import PlaygroundQueryRequest

            response = None
            try:
                response = await self._query.query(
                    owner_id,
                    PlaygroundQueryRequest(
                        question=body.question,
                        scope=body.scope,
                        document_ids=body.document_ids,
                        prompt_version_id=version_id,
                    ),
                )
                status = "completed"
                answer = response.answer
                citations = response.citations
                metrics: dict = {}
            except Exception as exc:
                status = "failed"
                answer = None
                citations = []
                metrics = {"error": type(exc).__name__}
            latency_ms = int((time.perf_counter() - started) * 1000)
            run = PlaygroundExperimentRun(
                id=uuid.uuid4(),
                experiment_id=experiment_id,
                owner_id=owner_id,
                prompt_version_id=version_id,
                prompt_snapshot=version.template_text,
                model_name=response.model_name if response else None,
                generation_config={},
                retrieval_scope=body.scope.value,
                document_ids=[str(d) for d in body.document_ids],
                input_question=body.question,
                answer=answer,
                citations=citations,
                metrics=metrics,
                token_usage=response.token_usage if response else {},
                estimated_cost_usd=(
                    (response.token_usage.get("estimated_cost_usd") if response else None)
                ),
                latency_ms=latency_ms,
                status=status,
            )
            async with self._session_factory() as session, session.begin():
                PlaygroundExperimentRepository(session).add_run(run)
            runs.append(run)
        return runs

    async def compare_runs(
        self,
        owner_id: uuid.UUID,
        experiment_id: uuid.UUID,
        run_ids: list[uuid.UUID],
    ) -> list[PlaygroundExperimentRun]:
        await self.get(owner_id, experiment_id)
        async with self._session_factory() as session:
            runs = await PlaygroundExperimentRepository(session).get_runs_by_ids(
                experiment_id, run_ids
            )
        if len(runs) != len(run_ids):
            raise ValidationFailedError("One or more run IDs are invalid for this experiment.")
        return runs
