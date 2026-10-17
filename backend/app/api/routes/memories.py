from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.dependencies.auth import PrincipalDep
from app.dependencies.services import MemoryServiceDep
from app.schemas.memory import MemoryList

router = APIRouter()

_ERRORS = {
    401: {"description": "Not authenticated"},
    404: {"description": "Memory not found (or owned by someone else)"},
}


@router.get("", response_model=MemoryList, responses=_ERRORS)
async def list_memories(
    principal: PrincipalDep,
    memory_service: MemoryServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MemoryList:
    items, total = await memory_service.list_memories(
        principal.owner_id, limit=limit, offset=offset
    )
    return MemoryList(items=items, total=total)


@router.patch(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_ERRORS,
)
async def deactivate_memory(
    memory_id: uuid.UUID,
    principal: PrincipalDep,
    memory_service: MemoryServiceDep,
) -> None:
    from app.core.exceptions import NotFoundError

    ok = await memory_service.deactivate(principal.owner_id, memory_id)
    if not ok:
        raise NotFoundError("Memory not found.")
