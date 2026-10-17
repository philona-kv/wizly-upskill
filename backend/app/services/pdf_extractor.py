"""PDF validation and page-by-page text extraction (PyMuPDF).

All functions here are synchronous and CPU-bound; call them via ``asyncio.to_thread``
from async code. Error messages are safe to show to users and never contain
document content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.exceptions import PdfExtractionError

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
# Some producers emit a few bytes of junk before the header; PDF readers accept it.
_MAGIC_SEARCH_WINDOW = 1024
MIN_TEXT_CHARS = 20


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    page_count: int
    pages: list[PageText]

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def has_pdf_signature(head: bytes) -> bool:
    return PDF_MAGIC in head[:_MAGIC_SEARCH_WINDOW]


def _open(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises a variety of types for broken files
        raise PdfExtractionError("The file is not a readable PDF (it may be corrupted).") from exc
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise PdfExtractionError("Password-protected or encrypted PDFs are not supported.")
    if doc.page_count == 0:
        doc.close()
        raise PdfExtractionError("The PDF has no pages.")
    return doc


def inspect_pdf(path: Path) -> int:
    """Validate that ``path`` is an openable, unencrypted PDF with pages. Returns page count."""
    doc = _open(path)
    try:
        return doc.page_count
    finally:
        doc.close()


_WS_RUN = re.compile(r"[ \t\f\v\u00a0]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [_WS_RUN.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text page by page, keeping 1-based page numbers.

    Pages are processed one at a time so only one page's text structures are alive
    at once. Raises ``PdfExtractionError`` if the document has no usable text
    (e.g. a scanned PDF - OCR is out of scope).
    """
    doc = _open(path)
    try:
        pages: list[PageText] = []
        for index in range(doc.page_count):
            try:
                page = doc.load_page(index)
                raw = page.get_text("text", sort=True)
            except Exception as exc:
                raise PdfExtractionError(
                    f"Could not read text from page {index + 1}; the PDF may be damaged."
                ) from exc
            pages.append(PageText(page_number=index + 1, text=_normalise(raw)))
        result = ExtractedDocument(page_count=doc.page_count, pages=pages)
    finally:
        doc.close()

    if result.total_chars < MIN_TEXT_CHARS:
        raise PdfExtractionError(
            "No extractable text was found. Scanned/image-only PDFs need OCR, "
            "which is not supported yet."
        )
    logger.debug(
        "pdf_extracted", extra={"page_count": result.page_count, "chars": result.total_chars}
    )
    return result
