"""Shared test fixtures.

The default suite never touches real OpenAI or Supabase. Tests marked ``db`` need a
disposable PostgreSQL + pgvector database given by TEST_DATABASE_URL; they are skipped
when it is unset. ``DATABASE_URL`` is ignored by tests so a developer's real database
can never be used by accident.
"""

from __future__ import annotations

import os

# Must run before app modules read settings / backend/.env.
os.environ["APP_ENV"] = "test"
for _secret in ("DATABASE_URL", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ[_secret] = ""

from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def make_pdf(pages: list[str], path: Path | None = None) -> bytes:
    """Build a text PDF. Each list item becomes one page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=9)
    data = doc.tobytes()
    doc.close()
    if path is not None:
        path.write_bytes(data)
    return data


def make_image_only_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(100, 100, 400, 400), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    data = doc.tobytes()
    doc.close()
    return data


def make_pdf_with_table(rows: int = 3, cols: int = 3) -> bytes:
    """PDF page with a bordered grid table (PyMuPDF ``find_tables``)."""
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(50, 80, 50 + cols * 90, 80 + rows * 28)
    cell_w = rect.width / cols
    cell_h = rect.height / rows
    for i in range(cols + 1):
        x = rect.x0 + i * cell_w
        page.draw_line((x, rect.y0), (x, rect.y1))
    for j in range(rows + 1):
        y = rect.y0 + j * cell_h
        page.draw_line((rect.x0, y), (rect.x1, y))
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (rect.x0 + c * cell_w + 4, rect.y0 + r * cell_h + 18),
                f"Cell{r}{c}",
                fontsize=10,
            )
    data = doc.tobytes()
    doc.close()
    return data


def make_pdf_with_embedded_image(image_bytes: bytes | None = None) -> bytes:
    """Text page with a raster image inserted (for embedded-image OCR tests)."""
    stream = image_bytes or make_image_bytes("PNG", with_text=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 120), SAMPLE_PARAGRAPH, fontsize=9)
    page.insert_image(pymupdf.Rect(50, 150, 300, 350), stream=stream)
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return data


def make_image_bytes(fmt: str = "PNG", *, with_text: bool = False) -> bytes:
    """Build a tiny PNG or JPEG. ``with_text`` draws letters for OCR tests (mocked)."""
    image = Image.new("RGB", (160, 80), color=(255, 255, 255))
    if with_text:
        ImageDraw.Draw(image).text((8, 24), "Sample text for OCR", fill=(0, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


SAMPLE_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "Retrieval systems depend on careful chunking of source documents. "
    "Each chunk should keep track of the pages it came from. "
)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return make_pdf([SAMPLE_PARAGRAPH * 3, SAMPLE_PARAGRAPH * 2])
