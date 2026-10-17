"""Image validation (Pillow) and OCR text extraction (Tesseract via pytesseract).

Synchronous and CPU-bound; call via ``asyncio.to_thread``. Error messages are safe
to show to users and never contain document content. ``ocr_image`` is a module-level
function so tests can replace Tesseract without a system binary.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.exceptions import ImageExtractionError
from app.services.pdf_extractor import MIN_TEXT_CHARS, ExtractedDocument, PageText, _normalise

logger = logging.getLogger(__name__)

_ALLOWED_FORMATS = frozenset({"PNG", "JPEG"})


def ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR. Isolated so tests can monkeypatch it."""
    try:
        import pytesseract
    except ImportError as exc:
        raise ImageExtractionError(
            "Image OCR is not installed. Install pytesseract and the Tesseract binary."
        ) from exc
    try:
        return pytesseract.image_to_string(image) or ""
    except pytesseract.TesseractNotFoundError as exc:
        raise ImageExtractionError(
            "Tesseract OCR is not installed on the server, so images cannot be ingested."
        ) from exc
    except Exception as exc:
        raise ImageExtractionError("Could not extract text from the image.") from exc


def inspect_image(path: Path) -> None:
    """Validate that ``path`` is a readable PNG or JPEG. Does not run OCR."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            if img.format not in _ALLOWED_FORMATS:
                raise ImageExtractionError("Only PNG and JPEG images are supported.")
            if img.width < 1 or img.height < 1:
                raise ImageExtractionError("The image has no pixels.")
    except ImageExtractionError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageExtractionError(
            "The file is not a readable PNG or JPEG image (it may be corrupted)."
        ) from exc


def extract_image(path: Path) -> ExtractedDocument:
    """OCR the image as a single 1-based page.

    Images with no usable text fail with a clear message. OCR quality depends on
    Tesseract; this is not a substitute for a text-based PDF.
    """
    inspect_image(path)
    with Image.open(path) as img:
        img.load()
        rgb = img.convert("RGB")
        raw = ocr_image(rgb)
    text = _normalise(raw)
    if len(text) < MIN_TEXT_CHARS:
        raise ImageExtractionError(
            "No extractable text was found in the image. Photos without readable "
            "text cannot be ingested."
        )
    logger.debug("image_extracted", extra={"chars": len(text)})
    return ExtractedDocument(page_count=1, pages=[PageText(page_number=1, text=text)])
