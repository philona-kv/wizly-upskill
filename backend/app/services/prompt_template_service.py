from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import NotFoundError, ValidationFailedError
from app.db.models.playground import PromptTemplate, PromptTemplateVersion
from app.repositories.playground_repository import PromptTemplateRepository
from app.services.prompt_renderer import validate_template_text


class PromptTemplateService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_templates(
        self, owner_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[PromptTemplate], int]:
        async with self._session_factory() as session:
            return await PromptTemplateRepository(session).list_visible(
                owner_id, limit=limit, offset=offset
            )

    async def get_template(
        self, owner_id: uuid.UUID, template_id: uuid.UUID
    ) -> PromptTemplate:
        async with self._session_factory() as session:
            template = await PromptTemplateRepository(session).get_for_owner(
                template_id, owner_id
            )
            if template is None:
                raise NotFoundError("Prompt template not found.")
            return template

    async def create_template(
        self, owner_id: uuid.UUID, *, name: str, description: str | None, template_text: str
    ) -> PromptTemplate:
        validate_template_text(template_text)
        template_id = uuid.uuid4()
        version_id = uuid.uuid4()
        async with self._session_factory() as session, session.begin():
            repo = PromptTemplateRepository(session)
            template = PromptTemplate(
                id=template_id,
                owner_id=owner_id,
                name=name,
                description=description,
                current_version=1,
                is_system=False,
            )
            repo.add(template)
            repo.add_version(
                PromptTemplateVersion(
                    id=version_id,
                    template_id=template_id,
                    version=1,
                    template_text=template_text,
                    config={},
                    created_by=owner_id,
                )
            )
        return template

    async def create_version(
        self,
        owner_id: uuid.UUID,
        template_id: uuid.UUID,
        *,
        template_text: str,
    ) -> PromptTemplateVersion:
        validate_template_text(template_text)
        async with self._session_factory() as session, session.begin():
            repo = PromptTemplateRepository(session)
            template = await repo.get_for_owner(template_id, owner_id)
            if template is None or template.is_system:
                raise NotFoundError("Prompt template not found.")
            if template.owner_id != owner_id:
                raise NotFoundError("Prompt template not found.")
            new_version = template.current_version + 1
            version = PromptTemplateVersion(
                id=uuid.uuid4(),
                template_id=template_id,
                version=new_version,
                template_text=template_text,
                config={},
                created_by=owner_id,
            )
            repo.add_version(version)
            template.current_version = new_version
            await session.merge(template)
            return version

    async def get_version_for_owner(
        self, owner_id: uuid.UUID, version_id: uuid.UUID
    ) -> PromptTemplateVersion:
        async with self._session_factory() as session:
            repo = PromptTemplateRepository(session)
            version = await repo.get_version(version_id)
            if version is None:
                raise NotFoundError("Prompt version not found.")
            template = await repo.get_for_owner(version.template_id, owner_id)
            if template is None:
                raise NotFoundError("Prompt version not found.")
            return version

    async def duplicate_template(
        self, owner_id: uuid.UUID, template_id: uuid.UUID
    ) -> PromptTemplate:
        async with self._session_factory() as session:
            repo = PromptTemplateRepository(session)
            source = await repo.get_for_owner(template_id, owner_id)
            if source is None:
                raise NotFoundError("Prompt template not found.")
            version = await repo.get_version_for_template(template_id, source.current_version)
            if version is None:
                raise ValidationFailedError("Template has no version to duplicate.")
            text = version.template_text
            name = f"{source.name} (copy)"
            description = source.description
        return await self.create_template(
            owner_id, name=name, description=description, template_text=text
        )

    async def list_versions(
        self, owner_id: uuid.UUID, template_id: uuid.UUID
    ) -> list[PromptTemplateVersion]:
        async with self._session_factory() as session:
            repo = PromptTemplateRepository(session)
            if await repo.get_for_owner(template_id, owner_id) is None:
                raise NotFoundError("Prompt template not found.")
            return await repo.list_versions(template_id)
