from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.core.exceptions import NotFoundError
from app.dependencies.auth import PrincipalDep
from app.dependencies.database import DbSessionDep
from app.dependencies.services import ChatServiceDep, SummaryServiceDep, WorkflowExecutionServiceDep
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversation import (
    ChatTurnResponse,
    ConversationCreate,
    ConversationList,
    ConversationRead,
    MessageCreate,
    MessageList,
    message_read,
)
from app.schemas.summary import SummaryRead
from app.schemas.workflow import WorkflowExecutionRead

router = APIRouter()

_ERRORS = {
    401: {"description": "Not authenticated"},
    404: {"description": "Conversation not found (or owned by someone else)"},
}


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ConversationRead)
async def create_conversation(
    principal: PrincipalDep,
    service: ChatServiceDep,
    body: ConversationCreate = ConversationCreate(),
) -> ConversationRead:
    conversation = await service.create_conversation(principal.owner_id, body.title)
    return ConversationRead.model_validate(conversation)


@router.get("", response_model=ConversationList, responses=_ERRORS)
async def list_conversations(
    principal: PrincipalDep,
    session: DbSessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationList:
    items, total = await ConversationRepository(session).list_for_owner(
        principal.owner_id, limit=limit, offset=offset
    )
    return ConversationList(
        items=[ConversationRead.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


async def _owned(session: DbSessionDep, principal: PrincipalDep, conversation_id: uuid.UUID):
    conversation = await ConversationRepository(session).get_for_owner(
        conversation_id, principal.owner_id
    )
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return conversation


@router.get("/{conversation_id}", response_model=ConversationRead, responses=_ERRORS)
async def get_conversation(
    conversation_id: uuid.UUID, principal: PrincipalDep, session: DbSessionDep
) -> ConversationRead:
    return ConversationRead.model_validate(await _owned(session, principal, conversation_id))


@router.get(
    "/{conversation_id}/messages",
    response_model=MessageList,
    responses=_ERRORS,
)
async def list_messages(
    conversation_id: uuid.UUID,
    principal: PrincipalDep,
    session: DbSessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MessageList:
    await _owned(session, principal, conversation_id)
    items, total = await ConversationRepository(session).list_messages(
        conversation_id, limit=limit, offset=offset
    )
    return MessageList(
        items=[message_read(m) for m in items], total=total, limit=limit, offset=offset
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=ChatTurnResponse,
    responses={
        **_ERRORS,
        502: {"description": "Embedding or generation failed after saving the user message"},
    },
)
async def send_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    principal: PrincipalDep,
    service: ChatServiceDep,
) -> ChatTurnResponse:
    return await service.send_message(
        principal.owner_id, conversation_id, body.content, body.client_message_id
    )


@router.get(
    "/{conversation_id}/summary",
    response_model=SummaryRead,
    responses={**_ERRORS, 404: {"description": "No summary yet"}},
)
async def get_conversation_summary(
    conversation_id: uuid.UUID,
    principal: PrincipalDep,
    session: DbSessionDep,
    summary_service: SummaryServiceDep,
) -> SummaryRead:
    await _owned(session, principal, conversation_id)
    summary = await summary_service.get_summary(principal.owner_id, conversation_id)
    if summary is None:
        raise NotFoundError("Conversation summary not found.")
    return summary


@router.get(
    "/{conversation_id}/workflows/{workflow_id}",
    response_model=WorkflowExecutionRead,
    responses=_ERRORS,
)
async def get_workflow_execution(
    conversation_id: uuid.UUID,
    workflow_id: uuid.UUID,
    principal: PrincipalDep,
    session: DbSessionDep,
    execution_service: WorkflowExecutionServiceDep,
) -> WorkflowExecutionRead:
    await _owned(session, principal, conversation_id)
    row = await execution_service.get(workflow_id, principal.owner_id)
    if row is None or row.conversation_id != conversation_id:
        raise NotFoundError("Workflow execution not found.")
    return row


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_ERRORS,
)
async def delete_conversation(
    conversation_id: uuid.UUID, principal: PrincipalDep, session: DbSessionDep
) -> Response:
    deleted = await ConversationRepository(session).delete(conversation_id, principal.owner_id)
    if not deleted:
        raise NotFoundError("Conversation not found.")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
