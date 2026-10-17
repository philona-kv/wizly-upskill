from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status

from app.core.exceptions import NotFoundError
from app.db.models import DocumentStatus, DocumentVisibility
from app.dependencies.auth import PrincipalDep
from app.dependencies.database import DbSessionDep
from app.dependencies.services import (
    DocumentPromotionServiceDep,
    DocumentServiceDep,
    IngestionServiceDep,
    JobRunnerDep,
    PlaygroundEvaluationServiceDep,
    PlaygroundExperimentServiceDep,
    PlaygroundQueryServiceDep,
    PromptTemplateServiceDep,
)
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentAccepted, DocumentRead, DocumentStatusRead
from app.schemas.playground import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    ExperimentCreate,
    ExperimentRunRead,
    ExperimentRunRequest,
    PlaygroundQueryRequest,
    PlaygroundQueryResponse,
    PromptTemplateCreate,
    PromptTemplateVersionCreate,
    PromptTemplateList,
    PromptTemplateRead,
    PromptTemplateVersionRead,
)
from app.services.document_service import IncomingFile

router = APIRouter()


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentAccepted)
async def upload_playground_document(
    response: Response,
    principal: PrincipalDep,
    service: DocumentServiceDep,
    runner: JobRunnerDep,
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form(max_length=255)] = None,
) -> DocumentAccepted:
    try:
        result = await service.upload(
            principal,
            IncomingFile(filename=file.filename, content_type=file.content_type, stream=file.file),
            visibility=DocumentVisibility.PRIVATE,
            title=title,
        )
    finally:
        await file.close()
    document = result.document
    if not result.duplicate:
        await runner.enqueue(document.id)
    response.status_code = status.HTTP_200_OK if result.duplicate else status.HTTP_202_ACCEPTED
    return DocumentAccepted(
        document_id=document.id,
        status=DocumentStatus(document.status),
        message="Upload accepted." if not result.duplicate else "Duplicate file.",
        duplicate=result.duplicate,
        document=DocumentRead.model_validate(document),
    )


@router.get("/documents")
async def list_playground_documents(
    principal: PrincipalDep,
    session: DbSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    visibility: Annotated[DocumentVisibility | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
) -> dict:
    items, total = await DocumentRepository(session).list_for_owner(
        principal.owner_id,
        limit=limit,
        offset=offset,
        status=status_filter,
        visibility=visibility,
        search=search,
    )
    return {
        "items": [DocumentRead.model_validate(d) for d in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_playground_document(
    document_id: uuid.UUID, principal: PrincipalDep, session: DbSessionDep
) -> DocumentRead:
    doc = await DocumentRepository(session).get_for_owner(document_id, principal.owner_id)
    if doc is None:
        raise NotFoundError("Document not found.")
    return DocumentRead.model_validate(doc)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_playground_document(
    document_id: uuid.UUID, principal: PrincipalDep, service: DocumentServiceDep
) -> Response:
    await service.delete(principal, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/documents/{document_id}/retry", response_model=DocumentStatusRead)
async def retry_playground_document(
    document_id: uuid.UUID,
    principal: PrincipalDep,
    service: DocumentServiceDep,
    ingestion: IngestionServiceDep,
    runner: JobRunnerDep,
) -> DocumentStatusRead:
    document = await service.prepare_retry(principal, document_id, ingestion.stale_before())
    await runner.enqueue(document.id)
    return DocumentStatusRead.model_validate(document)


@router.post("/documents/{document_id}/promote", response_model=DocumentRead)
async def promote_document(
    document_id: uuid.UUID,
    request: Request,
    principal: PrincipalDep,
    promotion: DocumentPromotionServiceDep,
) -> DocumentRead:
    request_id = request.headers.get("x-request-id")
    document = await promotion.promote_to_global(
        principal.owner_id, document_id, request_id=request_id
    )
    return DocumentRead.model_validate(document)


@router.post("/query", response_model=PlaygroundQueryResponse)
async def playground_query(
    body: PlaygroundQueryRequest,
    principal: PrincipalDep,
    service: PlaygroundQueryServiceDep,
) -> PlaygroundQueryResponse:
    return await service.query(principal.owner_id, body)


@router.get("/prompts", response_model=PromptTemplateList)
async def list_prompts(
    principal: PrincipalDep,
    prompts: PromptTemplateServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PromptTemplateList:
    items, total = await prompts.list_templates(principal.owner_id, limit=limit, offset=offset)
    return PromptTemplateList(
        items=[PromptTemplateRead.model_validate(t) for t in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/prompts", response_model=PromptTemplateRead, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    body: PromptTemplateCreate,
    principal: PrincipalDep,
    prompts: PromptTemplateServiceDep,
) -> PromptTemplateRead:
    template = await prompts.create_template(
        principal.owner_id,
        name=body.name,
        description=body.description,
        template_text=body.template_text,
    )
    return PromptTemplateRead.model_validate(template)


@router.get("/prompts/{prompt_id}", response_model=PromptTemplateRead)
async def get_prompt(
    prompt_id: uuid.UUID, principal: PrincipalDep, prompts: PromptTemplateServiceDep
) -> PromptTemplateRead:
    template = await prompts.get_template(principal.owner_id, prompt_id)
    return PromptTemplateRead.model_validate(template)


@router.get("/prompts/{prompt_id}/versions", response_model=list[PromptTemplateVersionRead])
async def list_prompt_versions(
    prompt_id: uuid.UUID, principal: PrincipalDep, prompts: PromptTemplateServiceDep
) -> list[PromptTemplateVersionRead]:
    versions = await prompts.list_versions(principal.owner_id, prompt_id)
    return [PromptTemplateVersionRead.model_validate(v) for v in versions]


@router.post(
    "/prompts/{prompt_id}/versions",
    response_model=PromptTemplateVersionRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_prompt_version(
    prompt_id: uuid.UUID,
    body: PromptTemplateVersionCreate,
    principal: PrincipalDep,
    prompts: PromptTemplateServiceDep,
) -> PromptTemplateVersionRead:
    version = await prompts.create_version(
        principal.owner_id, prompt_id, template_text=body.template_text
    )
    return PromptTemplateVersionRead.model_validate(version)


@router.post(
    "/prompts/{prompt_id}/duplicate",
    response_model=PromptTemplateRead,
    status_code=status.HTTP_201_CREATED,
)
async def duplicate_prompt(
    prompt_id: uuid.UUID, principal: PrincipalDep, prompts: PromptTemplateServiceDep
) -> PromptTemplateRead:
    template = await prompts.duplicate_template(principal.owner_id, prompt_id)
    return PromptTemplateRead.model_validate(template)


@router.post("/experiments", status_code=status.HTTP_201_CREATED)
async def create_experiment(
    body: ExperimentCreate,
    principal: PrincipalDep,
    experiments: PlaygroundExperimentServiceDep,
) -> dict:
    exp = await experiments.create(principal.owner_id, body.name)
    return {"id": str(exp.id), "name": exp.name, "status": exp.status}


@router.get("/experiments")
async def list_experiments(
    principal: PrincipalDep,
    experiments: PlaygroundExperimentServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    items, total = await experiments.list_experiments(
        principal.owner_id, limit=limit, offset=offset
    )
    return {
        "items": [{"id": str(e.id), "name": e.name, "status": e.status} for e in items],
        "total": total,
    }


@router.post("/experiments/{experiment_id}/run", response_model=list[ExperimentRunRead])
async def run_experiment(
    experiment_id: uuid.UUID,
    body: ExperimentRunRequest,
    principal: PrincipalDep,
    experiments: PlaygroundExperimentServiceDep,
) -> list[ExperimentRunRead]:
    runs = await experiments.run_experiment(principal.owner_id, experiment_id, body)
    return [ExperimentRunRead.model_validate(r) for r in runs]


@router.get("/experiments/{experiment_id}/compare")
async def compare_experiment_runs(
    experiment_id: uuid.UUID,
    principal: PrincipalDep,
    experiments: PlaygroundExperimentServiceDep,
    run_ids: Annotated[list[uuid.UUID], Query()],
) -> list[ExperimentRunRead]:
    runs = await experiments.compare_runs(principal.owner_id, experiment_id, run_ids)
    return [ExperimentRunRead.model_validate(r) for r in runs]


@router.post("/evaluations/run", response_model=EvaluationRunResponse)
async def run_evaluation(
    body: EvaluationRunRequest,
    principal: PrincipalDep,
    evaluation: PlaygroundEvaluationServiceDep,
) -> EvaluationRunResponse:
    return await evaluation.run(principal.owner_id, body)
