import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.workflow_execution_service import WorkflowExecutionService, make_thread_id


def test_thread_id_format() -> None:
    owner = uuid.UUID("00000000-0000-0000-0000-000000000001")
    conv = uuid.UUID("00000000-0000-0000-0000-000000000002")
    assert make_thread_id(owner, conv) == f"{owner}:{conv}"


@pytest.mark.asyncio
async def test_start_adds_running_row() -> None:
    session = AsyncMock()
    session.begin = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()))
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock()
    service = WorkflowExecutionService(factory)
    execution_id = uuid.uuid4()
    with patch("app.services.workflow_execution_service.WorkflowExecutionRepository") as repo_cls:
        repo = repo_cls.return_value
        await service.start(
            execution_id,
            owner_id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            user_message_id=uuid.uuid4(),
            client_message_id=uuid.uuid4(),
            graph_version="twin-v4",
        )
        repo.add.assert_called_once()
