from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.observability.llm_usage import clear_usage_records, usage_context, UsageContext
from app.schemas.playground import (
    PlaygroundQueryRequest,
    PlaygroundQueryResponse,
    PlaygroundRetrievalScope,
    RetrievedEvidenceRead,
)
from app.services.playground_document_access import PlaygroundDocumentAccessService
from app.services.playground_workflow_runner import PlaygroundWorkflowInput, PlaygroundWorkflowRunner
from app.services.prompt_template_service import PromptTemplateService
from app.workflows.schemas import EvidenceItem, EvidenceSourceType


class PlaygroundQueryService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        workflow: PlaygroundWorkflowRunner,
        prompt_service: PromptTemplateService,
    ) -> None:
        self._session_factory = session_factory
        self._workflow = workflow
        self._prompts = prompt_service
        self._access = PlaygroundDocumentAccessService()

    async def query(
        self, owner_id: uuid.UUID, body: PlaygroundQueryRequest
    ) -> PlaygroundQueryResponse:
        async with self._session_factory() as session:
            await self._access.validate_scope(
                session, owner_id, body.scope, body.document_ids
            )
        prompt_text: str | None = None
        if body.prompt_version_id is not None:
            version = await self._prompts.get_version_for_owner(
                owner_id, body.prompt_version_id
            )
            prompt_text = version.template_text

        request_id = uuid.uuid4()
        clear_usage_records()
        with usage_context(UsageContext(request_id=str(request_id))):
            output, final, usage = await self._workflow.run(
                PlaygroundWorkflowInput(
                    request_id=request_id,
                    owner_id=owner_id,
                    original_question=body.question,
                    scope=body.scope,
                    document_ids=list(body.document_ids),
                    prompt_template_text=prompt_text,
                    include_conversation_history=body.include_conversation_history,
                    conversation_id=body.conversation_id,
                )
            )

        doc_evidence = []
        for raw in final.get("document_evidence") or []:
            item = EvidenceItem.model_validate(raw)
            if item.source_type != EvidenceSourceType.DOCUMENT:
                continue
            doc_evidence.append(
                RetrievedEvidenceRead(
                    evidence_id=item.evidence_id,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    document_title=item.document_title,
                    original_filename=item.original_filename,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    retrieval_score=item.retrieval_score,
                    excerpt=(item.excerpt or item.content or "")[:500],
                )
            )

        meta = output.operational_metadata.model_dump(mode="json") if output.operational_metadata else None
        if meta is not None:
            meta["retrieval_scope"] = body.scope.value

        return PlaygroundQueryResponse(
            status=output.status.value,
            answer=output.answer,
            citations=[c.model_dump(mode="json") for c in output.citations],
            retrieval_scope=body.scope,
            document_evidence=doc_evidence,
            operational_metadata=meta,
            token_usage=usage,
            model_name=output.model_name,
        )
