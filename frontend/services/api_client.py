"""HTTP client for the AI Twin backend. No Streamlit imports - keep it unit-testable.

The frontend holds no secrets: it only knows the backend base URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"
TERMINAL_STATUSES = frozenset({"ready", "failed"})

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _content_type_for(filename: str) -> str:
    suffix = ""
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
    return _CONTENT_TYPES.get(suffix, "application/octet-stream")


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    document_title: str
    original_filename: str
    page_start: int
    page_end: int
    excerpt: str | None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Citation:
        return cls(
            citation_id=str(data.get("citation_id") or ""),
            document_title=str(data.get("document_title") or ""),
            original_filename=str(data.get("original_filename") or ""),
            page_start=int(data.get("page_start") or 0),
            page_end=int(data.get("page_end") or 0),
            excerpt=data.get("excerpt"),
        )

    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    id: str
    role: str
    content: str
    citations: list[Citation]
    created_at: datetime | None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ChatMessage:
        raw = data.get("citations") or []
        citations = [Citation.from_api(item) for item in raw] if isinstance(raw, list) else []
        return cls(
            id=str(data["id"]),
            role=str(data["role"]),
            content=str(data["content"]),
            citations=citations,
            created_at=_parse_dt(data.get("created_at")),
        )


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    title: str
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "Untitled"),
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
        )


@dataclass(frozen=True, slots=True)
class ConversationPage:
    items: list[Conversation]
    total: int


@dataclass(frozen=True, slots=True)
class ChatTurn:
    conversation_id: str
    user: ChatMessage
    assistant: ChatMessage
    operational_metadata: dict[str, Any] | None = None


class ApiError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, code: str = "error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

    @property
    def is_unavailable(self) -> bool:
        return self.code == "backend_unavailable"


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    original_filename: str
    visibility: str
    status: str
    error_message: str | None
    page_count: int | None
    chunk_count: int
    file_size_bytes: int
    embedding_model: str | None
    ingestion_attempts: int
    created_at: datetime | None
    updated_at: datetime | None
    processed_at: datetime | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Document:
        try:
            return cls(
                id=str(data["id"]),
                title=str(data.get("title") or ""),
                original_filename=str(data.get("original_filename") or ""),
                visibility=str(data["visibility"]),
                status=str(data["status"]),
                error_message=data.get("error_message"),
                page_count=data.get("page_count"),
                chunk_count=int(data.get("chunk_count") or 0),
                file_size_bytes=int(data.get("file_size_bytes") or 0),
                embedding_model=data.get("embedding_model"),
                ingestion_attempts=int(data.get("ingestion_attempts") or 0),
                created_at=_parse_dt(data.get("created_at")),
                updated_at=_parse_dt(data.get("updated_at")),
                processed_at=_parse_dt(data.get("processed_at")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(
                "Backend returned an unexpected document format.", code="invalid_response"
            ) from exc


@dataclass(frozen=True, slots=True)
class DocumentStatus:
    id: str
    status: str
    error_message: str | None
    page_count: int | None
    chunk_count: int

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class DocumentPage:
    items: list[Document]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class UploadResult:
    document: Document
    duplicate: bool
    message: str


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        upload_timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1",
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )
        self._upload_timeout = httpx.Timeout(upload_timeout, connect=5.0)

    @classmethod
    def from_env(cls) -> ApiClient:
        return cls(
            os.environ.get("STREAMLIT_API_URL", DEFAULT_API_URL),
            timeout=float(os.environ.get("STREAMLIT_API_TIMEOUT_SECONDS", "30")),
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(
                "The backend did not respond in time.", code="backend_unavailable"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(
                "Cannot reach the backend. Is the API running?", code="backend_unavailable"
            ) from exc
        if response.is_error:
            raise self._error_from(response)
        return response

    @staticmethod
    def _error_from(response: httpx.Response) -> ApiError:
        code, message = "http_error", f"Request failed with status {response.status_code}."
        try:
            error = response.json().get("error") or {}
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
        except (ValueError, AttributeError):
            pass
        return ApiError(message, status_code=response.status_code, code=code)

    @staticmethod
    def _parse_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError(
                "Backend returned an invalid response.", code="invalid_response"
            ) from exc

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        data = ApiClient._parse_json(response)
        if not isinstance(data, dict):
            raise ApiError("Backend returned an invalid response.", code="invalid_response")
        return data

    # -- endpoints ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._json(self._request("GET", "/health"))

    def readiness(self) -> dict[str, Any]:
        try:
            return self._json(self._request("GET", "/health/ready"))
        except ApiError as exc:
            if exc.status_code == 503:
                return {"status": "not_ready", "checks": {}}
            raise

    def list_documents(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        visibility: str | None = None,
    ) -> DocumentPage:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if visibility:
            params["visibility"] = visibility
        data = self._json(self._request("GET", "/documents", params=params))
        try:
            return DocumentPage(
                items=[Document.from_api(item) for item in data["items"]],
                total=int(data["total"]),
                limit=int(data["limit"]),
                offset=int(data["offset"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(
                "Backend returned an invalid document list.", code="invalid_response"
            ) from exc

    def get_document(self, document_id: str) -> Document:
        return Document.from_api(self._json(self._request("GET", f"/documents/{document_id}")))

    def get_status(self, document_id: str) -> DocumentStatus:
        data = self._json(self._request("GET", f"/documents/{document_id}/status"))
        try:
            return DocumentStatus(
                id=str(data["id"]),
                status=str(data["status"]),
                error_message=data.get("error_message"),
                page_count=data.get("page_count"),
                chunk_count=int(data.get("chunk_count") or 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("Backend returned an invalid status.", code="invalid_response") from exc

    def upload_document(
        self, filename: str, content: bytes, *, visibility: str, title: str | None = None
    ) -> UploadResult:
        form = {"visibility": visibility}
        if title and title.strip():
            form["title"] = title.strip()
        response = self._request(
            "POST",
            "/documents",
            files={"file": (filename, content, _content_type_for(filename))},
            data=form,
            timeout=self._upload_timeout,
        )
        data = self._json(response)
        if "document" not in data:
            raise ApiError("Backend returned an invalid upload response.", code="invalid_response")
        return UploadResult(
            document=Document.from_api(data["document"]),
            duplicate=bool(data.get("duplicate")),
            message=str(data.get("message") or ""),
        )

    def retry_document(self, document_id: str) -> None:
        self._request("POST", f"/documents/{document_id}/retry")

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def create_conversation(self, title: str | None = None) -> Conversation:
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        data = self._json(self._request("POST", "/conversations", json=payload))
        return Conversation.from_api(data)

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> ConversationPage:
        data = self._json(
            self._request("GET", "/conversations", params={"limit": limit, "offset": offset})
        )
        try:
            return ConversationPage(
                items=[Conversation.from_api(item) for item in data["items"]],
                total=int(data["total"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(
                "Backend returned an invalid conversation list.", code="invalid_response"
            ) from exc

    def get_conversation(self, conversation_id: str) -> Conversation:
        return Conversation.from_api(
            self._json(self._request("GET", f"/conversations/{conversation_id}"))
        )

    def get_messages(self, conversation_id: str, *, limit: int = 100) -> list[ChatMessage]:
        data = self._json(
            self._request(
                "GET", f"/conversations/{conversation_id}/messages", params={"limit": limit}
            )
        )
        try:
            return [ChatMessage.from_api(item) for item in data["items"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("Backend returned invalid messages.", code="invalid_response") from exc

    def send_message(
        self, conversation_id: str, content: str, *, client_message_id: str | None = None
    ) -> ChatTurn:
        payload: dict[str, Any] = {"content": content}
        if client_message_id:
            payload["client_message_id"] = client_message_id
        data = self._json(
            self._request(
                "POST",
                f"/conversations/{conversation_id}/messages",
                json=payload,
                timeout=httpx.Timeout(120.0, connect=5.0),
            )
        )
        try:
            meta = data.get("operational_metadata")
            return ChatTurn(
                conversation_id=str(data["conversation_id"]),
                user=ChatMessage.from_api(data["user_message"]),
                assistant=ChatMessage.from_api(data["assistant_message"]),
                operational_metadata=meta if isinstance(meta, dict) else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(
                "Backend returned an invalid chat response.",
                code="invalid_response",
            ) from exc

    def delete_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/conversations/{conversation_id}")

    def get_conversation_summary(self, conversation_id: str) -> dict[str, Any]:
        try:
            return self._json(
                self._request("GET", f"/conversations/{conversation_id}/summary")
            )
        except ApiError as exc:
            # Older APIs returned 404 when no summary existed yet.
            if exc.status_code == 404:
                return {
                    "conversation_id": conversation_id,
                    "summary_text": None,
                    "version": 0,
                    "summarized_message_count": 0,
                }
            raise

    def list_memories(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return self._json(
            self._request("GET", "/memories", params={"limit": limit, "offset": offset})
        )

    def deactivate_memory(self, memory_id: str) -> None:
        self._request("PATCH", f"/memories/{memory_id}")

    # -- playground --------------------------------------------------------

    def list_playground_documents(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        visibility: str | None = None,
        search: str | None = None,
    ) -> DocumentPage:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if visibility:
            params["visibility"] = visibility
        if search:
            params["search"] = search
        data = self._json(self._request("GET", "/playground/documents", params=params))
        return DocumentPage(
            items=[Document.from_api(item) for item in data["items"]],
            total=int(data["total"]),
            limit=int(data["limit"]),
            offset=int(data["offset"]),
        )

    def upload_playground_document(
        self, filename: str, content: bytes, *, title: str | None = None
    ) -> UploadResult:
        form: dict[str, Any] = {}
        if title and title.strip():
            form["title"] = title.strip()
        response = self._request(
            "POST",
            "/playground/documents",
            files={"file": (filename, content, _content_type_for(filename))},
            data=form,
            timeout=self._upload_timeout,
        )
        data = self._json(response)
        return UploadResult(
            document=Document.from_api(data["document"]),
            duplicate=bool(data.get("duplicate")),
            message=str(data.get("message") or ""),
        )

    def promote_playground_document(self, document_id: str) -> Document:
        return Document.from_api(
            self._json(self._request("POST", f"/playground/documents/{document_id}/promote"))
        )

    def playground_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json(
            self._request(
                "POST",
                "/playground/query",
                json=payload,
                timeout=httpx.Timeout(120.0, connect=5.0),
            )
        )

    def list_prompt_templates(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return self._json(
            self._request("GET", "/playground/prompts", params={"limit": limit, "offset": offset})
        )

    def list_prompt_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        data = self._parse_json(
            self._request("GET", f"/playground/prompts/{prompt_id}/versions")
        )
        if not isinstance(data, list):
            raise ApiError("Backend returned an invalid response.", code="invalid_response")
        return data

    def run_playground_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json(
            self._request(
                "POST",
                "/playground/evaluations/run",
                json=payload,
                timeout=httpx.Timeout(300.0, connect=5.0),
            )
        )
