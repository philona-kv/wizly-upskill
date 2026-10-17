"""Supported upload types: PDF, PNG, and JPEG.

Kind is determined from the filename extension. A declared MIME type, when present
and not the generic ``application/octet-stream``, must match that kind. Magic-byte
checks happen separately after the file has been streamed to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePath

from app.core.exceptions import DocumentExtractionError, UnsupportedMediaError

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
_MAGIC_SEARCH_WINDOW = 1024


class FileKind(StrEnum):
    PDF = "pdf"
    PNG = "png"
    JPEG = "jpeg"


@dataclass(frozen=True, slots=True)
class ResolvedFileType:
    kind: FileKind
    suffix: str
    mime_type: str


_PDF_MIMES = frozenset(
    {"application/pdf", "application/x-pdf", "application/acrobat", "application/octet-stream"}
)
_PNG_MIMES = frozenset({"image/png", "application/octet-stream"})
_JPEG_MIMES = frozenset({"image/jpeg", "image/jpg", "application/octet-stream"})

_EXTENSIONS: dict[str, ResolvedFileType] = {
    ".pdf": ResolvedFileType(FileKind.PDF, ".pdf", "application/pdf"),
    ".png": ResolvedFileType(FileKind.PNG, ".png", "image/png"),
    ".jpg": ResolvedFileType(FileKind.JPEG, ".jpg", "image/jpeg"),
    ".jpeg": ResolvedFileType(FileKind.JPEG, ".jpg", "image/jpeg"),
}

_KIND_MIMES: dict[FileKind, frozenset[str]] = {
    FileKind.PDF: _PDF_MIMES,
    FileKind.PNG: _PNG_MIMES,
    FileKind.JPEG: _JPEG_MIMES,
}


def resolve_upload_type(filename: str, content_type: str | None) -> ResolvedFileType:
    suffix = PurePath(filename).suffix.lower()
    resolved = _EXTENSIONS.get(suffix)
    if resolved is None:
        raise UnsupportedMediaError("Only PDF, PNG, and JPEG files are supported.")
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared and declared not in _KIND_MIMES[resolved.kind]:
        raise UnsupportedMediaError(
            f"Unsupported content type '{declared}' for a {resolved.kind.value.upper()} file."
        )
    return resolved


def has_magic(kind: FileKind, head: bytes) -> bool:
    window = head[:_MAGIC_SEARCH_WINDOW]
    if kind is FileKind.PDF:
        return PDF_MAGIC in window
    if kind is FileKind.PNG:
        return head.startswith(PNG_MAGIC)
    if kind is FileKind.JPEG:
        return head.startswith(JPEG_MAGIC)
    return False


def kind_from_storage_path(storage_path: str) -> FileKind:
    suffix = Path(storage_path).suffix.lower()
    if suffix == ".pdf":
        return FileKind.PDF
    if suffix == ".png":
        return FileKind.PNG
    if suffix in {".jpg", ".jpeg"}:
        return FileKind.JPEG
    raise DocumentExtractionError("The stored file type is not supported.")


def invalid_content_code(kind: FileKind) -> str:
    return "invalid_pdf" if kind is FileKind.PDF else "invalid_image"
