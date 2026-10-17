from pathlib import Path

import pytest

from app.core.exceptions import PdfExtractionError
from app.services.pdf_extractor import extract_pdf, has_pdf_signature, inspect_pdf
from tests.conftest import SAMPLE_PARAGRAPH, make_encrypted_pdf, make_image_only_pdf, make_pdf


def _write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(data)
    return path


def test_valid_text_pdf(tmp_path: Path) -> None:
    path = _write(tmp_path, make_pdf([SAMPLE_PARAGRAPH]))
    result = extract_pdf(path)
    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert "quick brown fox" in result.pages[0].text


def test_multi_page_pdf_preserves_one_based_page_numbers(tmp_path: Path) -> None:
    path = _write(tmp_path, make_pdf(["Alpha page text here.", "", "Gamma page text here."]))
    result = extract_pdf(path)
    assert result.page_count == 3
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert "Alpha" in result.pages[0].text
    assert result.pages[1].text == ""
    assert "Gamma" in result.pages[2].text


def test_pdf_with_only_blank_pages_has_no_text(tmp_path: Path) -> None:
    path = _write(tmp_path, make_pdf(["", ""]))
    with pytest.raises(PdfExtractionError, match="No extractable text"):
        extract_pdf(path)


def test_image_only_pdf_reports_ocr_unsupported(tmp_path: Path) -> None:
    path = _write(tmp_path, make_image_only_pdf())
    with pytest.raises(PdfExtractionError, match="OCR"):
        extract_pdf(path)


def test_corrupted_pdf(tmp_path: Path) -> None:
    path = _write(tmp_path, b"%PDF-1.7\n this is not really a pdf \x00\x01\x02")
    with pytest.raises(PdfExtractionError):
        extract_pdf(path)


def test_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path, b"")
    with pytest.raises(PdfExtractionError):
        inspect_pdf(path)


def test_encrypted_pdf_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, make_encrypted_pdf())
    with pytest.raises(PdfExtractionError, match="encrypted"):
        inspect_pdf(path)


def test_inspect_returns_page_count(tmp_path: Path) -> None:
    assert inspect_pdf(_write(tmp_path, make_pdf(["a", "b"]))) == 2


def test_pdf_signature() -> None:
    assert has_pdf_signature(b"%PDF-1.4 ...")
    assert has_pdf_signature(b"\xef\xbb\xbf%PDF-1.4")
    assert not has_pdf_signature(b"PK\x03\x04 zip file")
