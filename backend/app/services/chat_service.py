"""Chat orchestration: persist user turn, run LangGraph workflow, persist assistant."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import (
    GenerationError,
    NotFoundError,
    RetrievalError,
    ValidationFailedError,
)
from app.db.models import Conversation, Message, MessageRole
from app.repositories.conversation_repository import (
    DEFAULT_CONVERSATION_TITLE,
    ConversationRepository,
    title_from_question,
)
from app.schemas.conversation import ChatTurnResponse, message_read
from app.services.chat_postprocess import ChatPostProcessService
from app.services.twin_workflow_runner import TwinWorkflowRunner
from app.services.workflow_execution_service import WorkflowExecutionService
from app.workflows.schemas import CompactMessage, WorkflowInput

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingTurn:
    user_message_id: uuid.UUID
    history: list[Message]
    done: ChatTurnResponse | None = None


class ChatService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        workflow: TwinWorkflowRunner,
        *,
        history_limit: int,
        enable_debug_metadata: bool = False,
        execution_service: WorkflowExecutionService | None = None,
        graph_version: str = "twin-v4",
        postprocess: ChatPostProcessService | None = None,
        schedule_postprocess: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._workflow = workflow
        self._history_limit = history_limit
        self._enable_debug_metadata = enable_debug_metadata
        self._execution = execution_service
        self._graph_version = graph_version
        self._postprocess = postprocess
        self._schedule_postprocess = schedule_postprocess

    async def create_conversation(self, owner_id: uuid.UUID, title: str | None) -> Conversation:
        conversation = Conversation(
            owner_id=owner_id,
            title=(title or DEFAULT_CONVERSATION_TITLE).strip()[:255]
            or DEFAULT_CONVERSATION_TITLE,
        )
        async with self._session_factory() as session, session.begin():
            ConversationRepository(session).add(conversation)
            await session.flush()
            await session.refresh(conversation)
            return conversation

    async def send_message(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        client_message_id: uuid.UUID | None,
    ) -> ChatTurnResponse:
        question = content.strip()
        if not question:
            raise ValidationFailedError("Message content is required.")

        pending = await self._ensure_user_message(
            owner_id, conversation_id, question, client_message_id
        )
        if pending.done is not None:
            return pending.done

        if client_message_id is not None and self._execution is not None:
            completed = await self._execution.find_completed_client(
                owner_id, conversation_id, client_message_id
            )
            if completed is not None:
                async with self._session_factory() as session:
                    repo = ConversationRepository(session)
                    user_message = await session.get(Message, pending.user_message_id)
                    if user_message is None:
                        raise NotFoundError("Conversation not found.")
                    assistant = await repo.next_assistant_after(
                        conversation_id, user_message.id, user_message.created_at
                    )
                    if assistant is not None:
                        return ChatTurnResponse(
                            conversation_id=conversation_id,
                            user_message=message_read(user_message),
                            assistant_message=message_read(assistant),
                        )

        history_for_model = [m for m in pending.history if m.id != pending.user_message_id]
        recent = [
            CompactMessage(id=m.id, role=m.role, content=m.content) for m in history_for_model
        ]
        request_id = uuid.uuid4()
        workflow_input = WorkflowInput(
            request_id=request_id,
            owner_id=owner_id,
            conversation_id=conversation_id,
            current_user_message_id=pending.user_message_id,
            original_question=question,
            recent_messages=recent,
            client_message_id=client_message_id,
            graph_version=self._graph_version,
        )
        if self._execution is not None:
            await self._execution.start(
                request_id,
                owner_id=owner_id,
                conversation_id=conversation_id,
                user_message_id=pending.user_message_id,
                client_message_id=client_message_id,
                graph_version=self._graph_version,
            )
        try:
            result = await self._workflow.run(workflow_input)
        except (GenerationError, RetrievalError) as exc:
            if self._execution is not None:
                code = "retrieval_error" if isinstance(exc, RetrievalError) else "generation_error"
                await self._execution.fail(request_id, owner_id, code)
            logger.warning(
                "chat_generation_failed", extra={"conversation_id": str(conversation_id)}
            )
            raise
        if self._execution is not None:
            await self._execution.complete(request_id, owner_id)

        answer_text = result.answer
        citations = [c.model_dump(mode="json") for c in result.citations]
        used_llm = result.used_llm
        model_name = result.model_name

        async with self._session_factory() as session, session.begin():
            repo = ConversationRepository(session)
            user_message = await session.get(Message, pending.user_message_id)
            if user_message is None:
                raise NotFoundError("Conversation not found.")
            if client_message_id is not None:
                assistant_existing = await repo.next_assistant_after(
                    conversation_id, user_message.id, user_message.created_at
                )
                if assistant_existing is not None:
                    return ChatTurnResponse(
                        conversation_id=conversation_id,
                        user_message=message_read(user_message),
                        assistant_message=message_read(assistant_existing),
                    )
            assistant = repo.add_message(
                conversation_id=conversation_id,
                owner_id=owner_id,
                role=MessageRole.ASSISTANT,
                content=answer_text,
                citations=citations if used_llm or citations else [],
                model_name=model_name,
            )
            await repo.touch(conversation_id)
            await session.flush()
            await session.refresh(assistant)

        logger.info(
            "chat_turn_complete",
            extra={
                "conversation_id": str(conversation_id),
                "citations": len(citations),
                "used_llm": used_llm,
                "workflow_status": result.status.value,
            },
        )
        response = ChatTurnResponse(
            conversation_id=conversation_id,
            user_message=message_read(user_message),
            assistant_message=message_read(assistant),
        )
        if self._enable_debug_metadata and result.operational_metadata is not None:
            response = response.model_copy(
                update={
                    "operational_metadata": result.operational_metadata.model_dump(mode="json")
                }
            )
        if self._schedule_postprocess and self._postprocess is not None:
            self._postprocess.schedule(owner_id, conversation_id)
        return response

    async def _ensure_user_message(
        self,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        question: str,
        client_message_id: uuid.UUID | None,
    ) -> _PendingTurn:
        try:
            async with self._session_factory() as session, session.begin():
                return await self._insert_or_load_user(
                    session, owner_id, conversation_id, question, client_message_id
                )
        except IntegrityError:
            if client_message_id is None:
                raise ValidationFailedError("Could not save the message; please retry.") from None
            async with self._session_factory() as session, session.begin():
                return await self._insert_or_load_user(
                    session, owner_id, conversation_id, question, client_message_id
                )

    async def _insert_or_load_user(
        self,
        session: AsyncSession,
        owner_id: uuid.UUID,
        conversation_id: uuid.UUID,
        question: str,
        client_message_id: uuid.UUID | None,
    ) -> _PendingTurn:
        repo = ConversationRepository(session)
        conversation = await repo.get_for_owner(conversation_id, owner_id)
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        if client_message_id is not None:
            existing = await repo.get_by_client_message_id(conversation_id, client_message_id)
            if existing is not None:
                assistant = await repo.next_assistant_after(
                    conversation_id, existing.id, existing.created_at
                )
                history = await repo.recent_messages(conversation_id, self._history_limit)
                if assistant is not None:
                    return _PendingTurn(
                        user_message_id=existing.id,
                        history=history,
                        done=ChatTurnResponse(
                            conversation_id=conversation_id,
                            user_message=message_read(existing),
                            assistant_message=message_read(assistant),
                        ),
                    )
                return _PendingTurn(user_message_id=existing.id, history=history)
        history = await repo.recent_messages(conversation_id, self._history_limit)
        user_message = repo.add_message(
            conversation_id=conversation_id,
            owner_id=owner_id,
            role=MessageRole.USER,
            content=question,
            client_message_id=client_message_id,
        )
        await session.flush()
        await repo.set_title_if_default(conversation_id, title_from_question(question))
        await repo.touch(conversation_id)
        return _PendingTurn(user_message_id=user_message.id, history=history)
