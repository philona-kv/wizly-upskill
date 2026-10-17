from pathlib import Path

import pytest

from app.core.exceptions import ImageExtractionError
from app.services.image_extractor import extract_image, inspect_image
from tests.conftest import SAMPLE_PARAGRAPH, make_image_bytes


def _write(tmp_path: Path, data: bytes, name: str = "img.png") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_inspect_valid_png_and_jpeg(tmp_path: Path) -> None:
    inspect_image(_write(tmp_path, make_image_bytes("PNG"), "a.png"))
    inspect_image(_write(tmp_path, make_image_bytes("JPEG"), "a.jpg"))


def test_inspect_rejects_truncated_and_non_image(tmp_path: Path) -> None:
    with pytest.raises(ImageExtractionError, match="readable PNG or JPEG"):
        inspect_image(_write(tmp_path, b"\x89PNG\r\n\x1a\nnot-an-image"))
    with pytest.raises(ImageExtractionError, match="readable PNG or JPEG"):
        inspect_image(_write(tmp_path, b"hello world", "x.png"))


def test_extract_uses_ocr_and_is_one_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.image_extractor.ocr_image", lambda _img: SAMPLE_PARAGRAPH * 2)
    result = extract_image(_write(tmp_path, make_image_bytes("PNG", with_text=True)))
    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert "quick brown fox" in result.pages[0].text


def test_extract_jpeg_same_as_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.image_extractor.ocr_image", lambda _img: SAMPLE_PARAGRAPH)
    result = extract_image(_write(tmp_path, make_image_bytes("JPEG"), "photo.jpeg"))
    assert result.page_count == 1
    assert result.pages[0].text


def test_image_without_ocr_text_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.image_extractor.ocr_image", lambda _img: "")
    with pytest.raises(ImageExtractionError, match="No extractable text"):
        extract_image(_write(tmp_path, make_image_bytes("PNG")))
