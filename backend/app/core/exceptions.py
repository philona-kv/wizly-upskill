"""Application error hierarchy and FastAPI handlers.

Every error returned to clients has the shape ``{"error": {"code": ..., "message": ...}}``.
Messages are written for clients; internal details stay in the logs.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import ConfigurationError

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        self.message = message or self.message
        self.code = code or self.code
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "Resource not found."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class ValidationFailedError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class UnsupportedMediaError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"


class PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "Authentication required."


class ServiceUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"
    message = "A required service is unavailable."


class StorageError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "storage_error"
    message = "File storage operation failed."


class StorageObjectNotFoundError(StorageError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "storage_object_not_found"
    message = "Stored file not found."


class DocumentExtractionError(AppError):
    """Raised when a document cannot be read or has no usable text. Safe for clients."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "extraction_error"


class PdfExtractionError(DocumentExtractionError):
    """Raised for unreadable PDFs. ``message`` is safe to show to users."""

    code = "invalid_pdf"


class ImageExtractionError(DocumentExtractionError):
    """Raised for unreadable images or images with no OCR-able text."""

    code = "invalid_image"


class EmbeddingError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "embedding_error"
    message = "Embedding generation failed."


class RetrievalError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "retrieval_error"
    message = "Document retrieval failed."


class GenerationError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "generation_error"
    message = "Answer generation failed."


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning("app_error", extra={"code": exc.code, "status": exc.status_code})
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(ConfigurationError)
    async def _config_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        logger.error("configuration_error", extra={"detail": str(exc)})
        return _error_response(503, "service_unavailable", "Server is not fully configured.")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _error_response(exc.status_code, "http_error", message)

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            f"{'.'.join(str(p) for p in err.get('loc', ()) if p != 'body')}: {err.get('msg')}"
            for err in exc.errors()
        ]
        return _error_response(422, "validation_error", "; ".join(problems) or "Invalid request.")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"exc_type": type(exc).__name__})
        return _error_response(500, "internal_error", "An unexpected error occurred.")
