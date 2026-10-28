"""PDF validation and page-by-page text extraction (PyMuPDF).

Text is extracted with ``get_text``. Pages with little text and embedded raster images
are supplemented with Tesseract OCR (full-page render and/or embedded images).

All functions here are synchronous and CPU-bound; call them via ``asyncio.to_thread``
from async code. Error messages are safe to show to users and never contain
document content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image

from app.core.exceptions import PdfExtractionError

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
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


@dataclass(frozen=True, slots=True)
class PdfExtractOptions:
    page_ocr_enabled: bool = True
    page_ocr_scale: float = 2.0
    page_ocr_min_text_chars: int = 20
    embedded_image_ocr_enabled: bool = True
    embedded_image_min_pixels: int = 4096
    max_embedded_images_per_page: int = 5
    table_extraction_enabled: bool = True


def has_pdf_signature(head: bytes) -> bool:
    return PDF_MAGIC in head[:_MAGIC_SEARCH_WINDOW]


def _open(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:
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


def _run_ocr(image: Image.Image) -> str:
    from app.services.image_extractor import ocr_image

    return _normalise(ocr_image(image.convert("RGB")))


def _ocr_page_render(page: pymupdf.Page, scale: float) -> str:
    matrix = pymupdf.Matrix(scale, scale)
    try:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    except Exception:
        logger.debug("pdf_page_render_failed", extra={"page": page.number + 1})
        return ""
    if pixmap.width < 1 or pixmap.height < 1:
        return ""
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return _run_ocr(image)


def _ocr_embedded_images(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    *,
    min_pixels: int,
    max_images: int,
) -> str:
    blocks: list[str] = []
    seen: set[int] = set()
    for info in page.get_images(full=True):
        xref = int(info[0])
        if xref in seen:
            continue
        seen.add(xref)
        if len(blocks) >= max_images:
            break
        try:
            extracted = doc.extract_image(xref)
        except Exception:
            continue
        width = int(extracted.get("width") or 0)
        height = int(extracted.get("height") or 0)
        if width * height < min_pixels:
            continue
        try:
            image = Image.open(BytesIO(extracted["image"]))
            image.load()
        except Exception:
            continue
        text = _run_ocr(image)
        if text:
            blocks.append(text)
    return _normalise("\n\n".join(blocks))


def _merge_page_text(*parts: str) -> str:
    return _normalise("\n\n".join(p for p in parts if p.strip()))


def _rows_to_markdown(rows: list[list[object]]) -> str:
    cells = [[str(cell or "").strip() for cell in row] for row in rows]
    if not cells or not any(any(c for c in row) for row in cells):
        return ""
    width = max(len(row) for row in cells)
    padded = [row + [""] * (width - len(row)) for row in cells]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_to_markdown(table: object) -> str:
    to_markdown = getattr(table, "to_markdown", None)
    if callable(to_markdown):
        try:
            md = str(to_markdown()).strip()
            if md:
                return md
        except Exception:
            pass
    extract = getattr(table, "extract", None)
    if callable(extract):
        try:
            return _rows_to_markdown(extract())
        except Exception:
            return ""
    return ""


def _extract_tables_markdown(page: pymupdf.Page) -> str:
    if not hasattr(page, "find_tables"):
        return ""
    try:
        finder = page.find_tables()
    except Exception:
        logger.debug("pdf_find_tables_failed", extra={"page": page.number + 1})
        return ""
    blocks: list[str] = []
    for table in getattr(finder, "tables", []) or []:
        md = _table_to_markdown(table)
        if md:
            blocks.append(md)
    return _normalise("\n\n".join(blocks))


def _extract_page(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    *,
    options: PdfExtractOptions,
) -> str:
    try:
        raw = page.get_text("text", sort=True)
    except Exception as exc:
        raise PdfExtractionError(
            f"Could not read text from page {page.number + 1}; the PDF may be damaged."
        ) from exc
    text = _normalise(raw)
    if options.table_extraction_enabled:
        tables_md = _extract_tables_markdown(page)
        if tables_md:
            text = _merge_page_text(text, f"Structured tables:\n{tables_md}")
    used_render_ocr = False

    if options.page_ocr_enabled and len(text) < options.page_ocr_min_text_chars:
        render_text = _ocr_page_render(page, options.page_ocr_scale)
        if render_text:
            text = _merge_page_text(text, render_text)
            used_render_ocr = True

    if (
        options.embedded_image_ocr_enabled
        and not used_render_ocr
        and len(text) >= options.page_ocr_min_text_chars
    ):
        embedded = _ocr_embedded_images(
            doc,
            page,
            min_pixels=options.embedded_image_min_pixels,
            max_images=options.max_embedded_images_per_page,
        )
        if embedded:
            text = _merge_page_text(text, embedded)

    return text


def extract_pdf(path: Path, options: PdfExtractOptions | None = None) -> ExtractedDocument:
    """Extract text page by page, keeping 1-based page numbers.

    Sparse or image-only pages use OCR when enabled in ``options``.
    """
    opts = options or PdfExtractOptions()
    doc = _open(path)
    try:
        pages: list[PageText] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = _extract_page(doc, page, options=opts)
            pages.append(PageText(page_number=index + 1, text=text))
        result = ExtractedDocument(page_count=doc.page_count, pages=pages)
    finally:
        doc.close()

    if result.total_chars < MIN_TEXT_CHARS:
        raise PdfExtractionError(
            "No extractable text was found. The PDF may be scanned or image-only without "
            "readable content, or OCR may not be installed."
        )
    logger.debug(
        "pdf_extracted", extra={"page_count": result.page_count, "chars": result.total_chars}
    )
    return result
