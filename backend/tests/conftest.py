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
