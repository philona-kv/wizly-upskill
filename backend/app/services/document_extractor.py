"""Dispatch inspection and extraction by uploaded file kind."""

from __future__ import annotations

from pathlib import Path

from app.services.file_types import FileKind
from app.services.image_extractor import extract_image, inspect_image
from app.services.pdf_extractor import ExtractedDocument, extract_pdf, inspect_pdf


def inspect_document(path: Path, kind: FileKind) -> None:
    if kind is FileKind.PDF:
        inspect_pdf(path)
        return
    inspect_image(path)


def extract_document(path: Path, kind: FileKind) -> ExtractedDocument:
    if kind is FileKind.PDF:
        return extract_pdf(path)
    return extract_image(path)
