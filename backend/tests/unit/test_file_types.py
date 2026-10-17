import pytest

from app.core.exceptions import DocumentExtractionError, UnsupportedMediaError
from app.services.file_types import (
    FileKind,
    has_magic,
    invalid_content_code,
    kind_from_storage_path,
    resolve_upload_type,
)


@pytest.mark.parametrize(
    ("filename", "content_type", "kind", "suffix", "mime"),
    [
        ("notes.pdf", "application/pdf", FileKind.PDF, ".pdf", "application/pdf"),
        ("scan.PNG", "image/png", FileKind.PNG, ".png", "image/png"),
        ("photo.jpg", "image/jpeg", FileKind.JPEG, ".jpg", "image/jpeg"),
        ("photo.JPEG", None, FileKind.JPEG, ".jpg", "image/jpeg"),
        ("a.pdf", "application/octet-stream", FileKind.PDF, ".pdf", "application/pdf"),
        ("a.png", "", FileKind.PNG, ".png", "image/png"),
    ],
)
def test_resolve_supported_types(filename, content_type, kind, suffix, mime) -> None:
    resolved = resolve_upload_type(filename, content_type)
    assert resolved.kind is kind
    assert resolved.suffix == suffix
    assert resolved.mime_type == mime


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("notes.txt", "text/plain"),
        ("image.gif", "image/gif"),
        ("doc.pdf", "image/png"),
        ("photo.png", "application/pdf"),
    ],
)
def test_resolve_rejects_unsupported(filename, content_type) -> None:
    with pytest.raises(UnsupportedMediaError):
        resolve_upload_type(filename, content_type)


def test_magic_bytes() -> None:
    assert has_magic(FileKind.PDF, b"%PDF-1.4 rest")
    assert has_magic(FileKind.PNG, b"\x89PNG\r\n\x1a\nrest")
    assert has_magic(FileKind.JPEG, b"\xff\xd8\xff\xe0rest")
    assert not has_magic(FileKind.PNG, b"%PDF-1.4")
    assert not has_magic(FileKind.JPEG, b"\x89PNG\r\n\x1a\n")


def test_kind_from_storage_path() -> None:
    assert kind_from_storage_path("owner/id.pdf") is FileKind.PDF
    assert kind_from_storage_path("owner/id.png") is FileKind.PNG
    assert kind_from_storage_path("owner/id.jpg") is FileKind.JPEG
    with pytest.raises(DocumentExtractionError):
        kind_from_storage_path("owner/id.gif")


def test_invalid_content_codes() -> None:
    assert invalid_content_code(FileKind.PDF) == "invalid_pdf"
    assert invalid_content_code(FileKind.PNG) == "invalid_image"
    assert invalid_content_code(FileKind.JPEG) == "invalid_image"
