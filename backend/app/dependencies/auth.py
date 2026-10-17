"""Owner resolution.

DEVELOPMENT ONLY: with ``AUTH_MODE=development`` every request is attributed to the
single configured ``DEV_OWNER_ID``. Nothing supplied by the client influences the
owner. This is NOT authentication and must not be exposed to untrusted networks.

To add real authentication, replace ``get_current_principal`` (e.g. verify a
Supabase JWT from the ``Authorization`` header and use its ``sub`` claim). All
routes and services depend only on ``Principal``, so nothing else changes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import AuthMode, Settings
from app.core.exceptions import UnauthorizedError


@dataclass(frozen=True, slots=True)
class Principal:
    owner_id: uuid.UUID
    is_development_identity: bool = False


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def get_current_principal(settings: SettingsDep) -> Principal:
    if settings.auth_mode == AuthMode.DEVELOPMENT:
        return Principal(owner_id=settings.dev_owner_id, is_development_identity=True)
    raise UnauthorizedError()


PrincipalDep = Annotated[Principal, Depends(get_current_principal)]
