"""Shared Streamlit helpers: the cached API client and one-shot flash messages."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from services.api_client import ApiClient

FlashLevel = Literal["success", "info", "warning", "error"]


# Bump when ApiClient methods change; otherwise Streamlit keeps a stale cached instance.
_API_CLIENT_CACHE_VERSION = 6


@st.cache_resource
def get_api_client(*, _cache_version: int = _API_CLIENT_CACHE_VERSION) -> ApiClient:
    del _cache_version
    return ApiClient.from_env()


def flash(level: FlashLevel, message: str) -> None:
    st.session_state.setdefault("_flash", []).append((level, message))


def render_flashes() -> None:
    for level, message in st.session_state.pop("_flash", []):
        getattr(st, level)(message)
