from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.playground import (
    DocumentPromotionAudit,
    PlaygroundExperiment,
    PlaygroundExperimentRun,
    PromptTemplate,
    PromptTemplateVersion,
)


class PromptTemplateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, template_id: uuid.UUID) -> PromptTemplate | None:
        return await self.session.get(PromptTemplate, template_id)

    async def get_for_owner(
        self, template_id: uuid.UUID, owner_id: uuid.UUID
    ) -> PromptTemplate | None:
        stmt = select(PromptTemplate).where(
            PromptTemplate.id == template_id,
            or_(PromptTemplate.is_system.is_(True), PromptTemplate.owner_id == owner_id),
        )
        return await self.session.scalar(stmt)

    async def list_visible(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[PromptTemplate], int]:
        cond = or_(PromptTemplate.is_system.is_(True), PromptTemplate.owner_id == owner_id)
        total = (
            await self.session.scalar(
                select(func.count()).select_from(PromptTemplate).where(cond)
            )
            or 0
        )
        stmt = (
            select(PromptTemplate)
            .where(cond)
            .order_by(PromptTemplate.is_system.desc(), PromptTemplate.name)
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.scalars(stmt)).all()), total

    def add(self, template: PromptTemplate) -> None:
        self.session.add(template)

    async def get_version(self, version_id: uuid.UUID) -> PromptTemplateVersion | None:
        return await self.session.get(PromptTemplateVersion, version_id)

    async def get_version_for_template(
        self, template_id: uuid.UUID, version: int
    ) -> PromptTemplateVersion | None:
        stmt = select(PromptTemplateVersion).where(
            PromptTemplateVersion.template_id == template_id,
            PromptTemplateVersion.version == version,
        )
        return await self.session.scalar(stmt)

    async def list_versions(self, template_id: uuid.UUID) -> list[PromptTemplateVersion]:
        stmt = (
            select(PromptTemplateVersion)
            .where(PromptTemplateVersion.template_id == template_id)
            .order_by(PromptTemplateVersion.version.desc())
        )
        return list((await self.session.scalars(stmt)).all())

    def add_version(self, version: PromptTemplateVersion) -> None:
        self.session.add(version)


class PlaygroundExperimentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_for_owner(
        self, experiment_id: uuid.UUID, owner_id: uuid.UUID
    ) -> PlaygroundExperiment | None:
        stmt = select(PlaygroundExperiment).where(
            PlaygroundExperiment.id == experiment_id,
            PlaygroundExperiment.owner_id == owner_id,
        )
        return await self.session.scalar(stmt)

    async def list_for_owner(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[list[PlaygroundExperiment], int]:
        cond = PlaygroundExperiment.owner_id == owner_id
        total = (
            await self.session.scalar(
                select(func.count()).select_from(PlaygroundExperiment).where(cond)
            )
            or 0
        )
        stmt = (
            select(PlaygroundExperiment)
            .where(cond)
            .order_by(PlaygroundExperiment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.scalars(stmt)).all()), total

    def add(self, experiment: PlaygroundExperiment) -> None:
        self.session.add(experiment)

    async def list_runs(self, experiment_id: uuid.UUID) -> list[PlaygroundExperimentRun]:
        stmt = (
            select(PlaygroundExperimentRun)
            .where(PlaygroundExperimentRun.experiment_id == experiment_id)
            .order_by(PlaygroundExperimentRun.created_at.desc())
        )
        return list((await self.session.scalars(stmt)).all())

    async def get_runs_by_ids(
        self, experiment_id: uuid.UUID, run_ids: Sequence[uuid.UUID]
    ) -> list[PlaygroundExperimentRun]:
        if not run_ids:
            return []
        stmt = select(PlaygroundExperimentRun).where(
            PlaygroundExperimentRun.experiment_id == experiment_id,
            PlaygroundExperimentRun.id.in_(list(run_ids)),
        )
        return list((await self.session.scalars(stmt)).all())

    def add_run(self, run: PlaygroundExperimentRun) -> None:
        self.session.add(run)


class DocumentPromotionAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, audit: DocumentPromotionAudit) -> None:
        self.session.add(audit)
