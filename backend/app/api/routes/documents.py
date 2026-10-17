from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from app.core.exceptions import NotFoundError
from app.db.models import DocumentStatus, DocumentVisibility
from app.dependencies.auth import PrincipalDep
from app.dependencies.database import DbSessionDep
from app.dependencies.services import DocumentServiceDep, IngestionServiceDep, JobRunnerDep
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentAccepted, DocumentList, DocumentRead, DocumentStatusRead
from app.services.document_service import IncomingFile

router = APIRouter()

_ERRORS = {
    401: {"description": "Not authenticated"},
    404: {"description": "Document not found (or owned by someone else)"},
}


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentAccepted,
    responses={
        200: {"description": "Duplicate of an existing document; nothing re-processed"},
        409: {"description": "Duplicate with conflicting visibility, or being deleted"},
        413: {"description": "File too large"},
        415: {"description": "Not a PDF, PNG, or JPEG"},
        422: {"description": "Malformed, empty, encrypted, or unreadable file"},
    },
)
async def upload_document(
    response: Response,
    principal: PrincipalDep,
    service: DocumentServiceDep,
    runner: JobRunnerDep,
    file: Annotated[UploadFile, File(description="PDF, PNG, or JPEG file")],
    visibility: Annotated[DocumentVisibility, Form()],
    title: Annotated[str | None, Form(max_length=255)] = None,
) -> DocumentAccepted:
    try:
        result = await service.upload(
            principal,
            IncomingFile(filename=file.filename, content_type=file.content_type, stream=file.file),
            visibility=visibility,
            title=title,
        )
    finally:
        await file.close()

    document = result.document
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
        message = "This file was already uploaded; returning the existing document."
    else:
        await runner.enqueue(document.id)
        message = "Upload accepted; ingestion has been scheduled."
    return DocumentAccepted(
        document_id=document.id,
        status=DocumentStatus(document.status),
        message=message,
        duplicate=result.duplicate,
        document=DocumentRead.model_validate(document),
    )


@router.get("", response_model=DocumentList, responses=_ERRORS)
async def list_documents(
    principal: PrincipalDep,
    session: DbSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    visibility: Annotated[DocumentVisibility | None, Query()] = None,
) -> DocumentList:
    items, total = await DocumentRepository(session).list_for_owner(
        principal.owner_id, limit=limit, offset=offset, status=status_filter, visibility=visibility
    )
    return DocumentList(
        items=[DocumentRead.model_validate(d) for d in items],
        total=total,
        limit=limit,
        offset=offset,
    )


async def _get_owned(session: DbSessionDep, principal: PrincipalDep, document_id: uuid.UUID):
    document = await DocumentRepository(session).get_for_owner(document_id, principal.owner_id)
    if document is None:
        # Same response whether the document is missing or belongs to another owner.
        raise NotFoundError("Document not found.")
    return document


@router.get("/{document_id}", response_model=DocumentRead, responses=_ERRORS)
async def get_document(
    document_id: uuid.UUID, principal: PrincipalDep, session: DbSessionDep
) -> DocumentRead:
    return DocumentRead.model_validate(await _get_owned(session, principal, document_id))


@router.get("/{document_id}/status", response_model=DocumentStatusRead, responses=_ERRORS)
async def get_document_status(
    document_id: uuid.UUID, principal: PrincipalDep, session: DbSessionDep
) -> DocumentStatusRead:
    return DocumentStatusRead.model_validate(await _get_owned(session, principal, document_id))


@router.post(
    "/{document_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentStatusRead,
    responses={**_ERRORS, 409: {"description": "Document is not in a retryable state"}},
)
async def retry_document(
    document_id: uuid.UUID,
    principal: PrincipalDep,
    service: DocumentServiceDep,
    ingestion: IngestionServiceDep,
    runner: JobRunnerDep,
) -> DocumentStatusRead:
    document = await service.prepare_retry(principal, document_id, ingestion.stale_before())
    await runner.enqueue(document.id)
    return DocumentStatusRead.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={**_ERRORS, 502: {"description": "Storage deletion failed; retry the request"}},
)
async def delete_document(
    document_id: uuid.UUID, principal: PrincipalDep, service: DocumentServiceDep
) -> Response:
    await service.delete(principal, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
