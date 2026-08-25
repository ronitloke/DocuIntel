"""Tests for Module 2 OCR and heuristic document structure extraction."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app
from app.services.ocr.tesseract_ocr import OCRResult, TesseractOCRService


def render_page_as_png(page: fitz.Page, dpi: int = 250) -> bytes:
    """Render one native page to an in-memory PNG."""

    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    finally:
        image.close()


def make_native_pdf() -> bytes:
    """Create native text with a heading, paragraph, and list item."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 80), "Document Heading", fontsize=24, fontname="hebo")
    page.insert_text(
        (72, 125),
        "This paragraph contains enough native text for the ingestion pipeline.",
        fontsize=11,
    )
    page.insert_text((72, 160), "- A native list item", fontsize=11)
    try:
        return document.tobytes()
    finally:
        document.close()


def make_image_only_pdf(text: str = "Scanned OCR integration text 12345") -> bytes:
    """Create an image-only PDF by rasterising a native source page."""

    source = fitz.open()
    source_page = source.new_page()
    source_page.insert_text((72, 130), text, fontsize=22)
    page_rect = source_page.rect
    image_bytes = render_page_as_png(source_page)
    source.close()

    scanned = fitz.open()
    scanned_page = scanned.new_page(width=page_rect.width, height=page_rect.height)
    scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
    try:
        return scanned.tobytes()
    finally:
        scanned.close()


def make_mixed_pdf() -> bytes:
    """Create a native first page and image-only second page."""

    document = fitz.open()
    native_page = document.new_page()
    native_page.insert_text((72, 100), "Native page remains native.", fontsize=18)

    source = fitz.open()
    source_page = source.new_page()
    source_page.insert_text((72, 100), "Scanned page goes through OCR.", fontsize=18)
    page_rect = source_page.rect
    image_bytes = render_page_as_png(source_page)
    source.close()

    scanned_page = document.new_page(width=page_rect.width, height=page_rect.height)
    scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
    try:
        return document.tobytes()
    finally:
        document.close()


def make_table_pdf() -> bytes:
    """Create a simple ruled native table for PyMuPDF table detection."""

    document = fitz.open()
    page = document.new_page(width=400, height=300)
    x_positions = [50, 200, 300]
    y_positions = [50, 100, 150]
    for x_position in x_positions:
        page.draw_line((x_position, y_positions[0]), (x_position, y_positions[-1]))
    for y_position in y_positions:
        page.draw_line((x_positions[0], y_position), (x_positions[-1], y_position))
    page.insert_text((60, 80), "Product", fontsize=11)
    page.insert_text((210, 80), "Qty", fontsize=11)
    page.insert_text((60, 130), "Laptop", fontsize=11)
    page.insert_text((210, 130), "3", fontsize=11)
    try:
        return document.tobytes()
    finally:
        document.close()


class FakeOCRService:
    """Deterministic OCR double for pipeline and failure tests."""

    def __init__(self, result: OCRResult | None = None, error: Exception | None = None) -> None:
        self.result = result or OCRResult(
            text="Recovered OCR text from the scanned page.",
            success=True,
            confidence=91.4,
        )
        self.error = error
        self.availability_calls = 0
        self.extraction_calls = 0

    def is_available(self) -> bool:
        """Report that the fake OCR engine is available."""

        self.availability_calls += 1
        return True

    def extract(self, image: Image.Image) -> OCRResult:
        """Return the configured result or raise the configured error."""

        self.extraction_calls += 1
        if self.error:
            raise self.error
        return self.result


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Provide an API client with an isolated upload directory."""

    with TestClient(create_app(storage_directory=tmp_path / "uploads")) as test_client:
        yield test_client


def upload_pdf(client: TestClient, content: bytes) -> dict:
    """Upload a generated PDF and return its JSON response."""

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("module2.pdf", BytesIO(content), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_native_pdf_skips_ocr_and_returns_layout_elements(tmp_path: Path) -> None:
    """Native pages stay native and produce heuristic structure elements."""

    fake_ocr = FakeOCRService()
    with TestClient(
        create_app(storage_directory=tmp_path / "uploads", ocr_service=fake_ocr)
    ) as client:
        payload = upload_pdf(client, make_native_pdf())

    page = payload["pages"][0]
    element_types = [element["element_type"] for element in page["layout_elements"]]
    assert fake_ocr.availability_calls == 0
    assert fake_ocr.extraction_calls == 0
    assert page["extraction_method"] == "native"
    assert page.get("ocr_applied", False) is False
    assert "heading" in element_types
    assert "paragraph" in element_types
    assert "list_item" in element_types
    assert payload["heading_count"] >= 1
    assert payload["layout_element_count"] >= 3


def test_image_only_pdf_uses_mocked_ocr(tmp_path: Path) -> None:
    """An image-only page is rendered and its OCR result is exposed."""

    fake_ocr = FakeOCRService()
    with TestClient(
        create_app(storage_directory=tmp_path / "uploads", ocr_service=fake_ocr)
    ) as client:
        payload = upload_pdf(client, make_image_only_pdf())

    page = payload["pages"][0]
    assert fake_ocr.extraction_calls == 1
    assert page["page_number"] == 1
    assert page["text"] == "Recovered OCR text from the scanned page."
    assert page["has_native_text"] is False
    assert page["needs_ocr"] is False
    assert page["ocr_applied"] is True
    assert page["ocr_success"] is True
    assert page["ocr_confidence"] == 91.4
    assert page["extraction_method"] == "ocr"
    assert payload["pages_processed_by_ocr"] == 1
    assert payload["ocr_failed_pages"] == 0
    assert payload["unresolved_ocr_pages"] == 0


def test_mixed_pdf_only_ocr_processes_scanned_page(tmp_path: Path) -> None:
    """Mixed PDFs preserve order and call OCR only for the image page."""

    fake_ocr = FakeOCRService()
    with TestClient(
        create_app(storage_directory=tmp_path / "uploads", ocr_service=fake_ocr)
    ) as client:
        payload = upload_pdf(client, make_mixed_pdf())

    pages = payload["pages"]
    assert fake_ocr.extraction_calls == 1
    assert [page["page_number"] for page in pages] == [1, 2]
    assert pages[0]["extraction_method"] == "native"
    assert pages[0].get("ocr_applied", False) is False
    assert pages[1]["extraction_method"] == "ocr"
    assert pages[1]["ocr_applied"] is True
    assert payload["pages_with_native_text"] == 1
    assert payload["pages_processed_by_ocr"] == 1


def test_ocr_failure_is_reported_as_unresolved(tmp_path: Path) -> None:
    """OCR exceptions do not become false successes or whole-document crashes."""

    fake_ocr = FakeOCRService(error=RuntimeError("simulated OCR failure"))
    with TestClient(
        create_app(storage_directory=tmp_path / "uploads", ocr_service=fake_ocr)
    ) as client:
        payload = upload_pdf(client, make_image_only_pdf())

    page = payload["pages"][0]
    assert page["ocr_applied"] is True
    assert page["ocr_success"] is False
    assert page["needs_ocr"] is True
    assert page["extraction_method"] == "ocr"
    assert payload["pages_processed_by_ocr"] == 1
    assert payload["ocr_failed_pages"] == 1
    assert payload["unresolved_ocr_pages"] == 1


def test_tesseract_resolution_uses_explicit_path() -> None:
    """The configured Windows executable is resolved and runnable."""

    command = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    service = TesseractOCRService(Settings(tesseract_cmd=command))

    assert service.resolved_command == command
    assert service.is_available() is True


def test_unavailable_tesseract_returns_controlled_result() -> None:
    """A missing executable produces a controlled OCR failure result."""

    service = TesseractOCRService(Settings(tesseract_cmd=r"C:\missing\tesseract.exe"))
    image = Image.new("RGB", (100, 100), "white")
    try:
        result = service.extract(image)
    finally:
        image.close()

    assert result.success is False
    assert result.confidence is None
    assert result.error == "Tesseract OCR is unavailable."


def test_native_table_is_returned_as_rows_and_layout_element(
    tmp_path: Path,
) -> None:
    """A simple ruled native table is detected without a dedicated ML model."""

    with TestClient(create_app(storage_directory=tmp_path / "uploads")) as client:
        payload = upload_pdf(client, make_table_pdf())

    assert payload["table_count"] == 1
    table = payload["pages"][0]["tables"][0]
    assert table["headers"] == ["Product", "Qty"]
    assert table["rows"] == [["Laptop", "3"]]
    assert any(
        element["element_type"] == "table"
        for element in payload["pages"][0]["layout_elements"]
    )


@pytest.mark.skipif(
    not TesseractOCRService().is_available(),
    reason="Tesseract is not available for the optional local OCR integration test",
)
def test_local_tesseract_recovers_image_only_text(tmp_path: Path) -> None:
    """The installed local Tesseract recovers readable text from an image-only PDF."""

    settings = Settings(ocr_render_dpi=300)
    with TestClient(
        create_app(settings=settings, storage_directory=tmp_path / "uploads")
    ) as client:
        payload = upload_pdf(client, make_image_only_pdf("Local Tesseract OCR check 67890"))

    page = payload["pages"][0]
    assert page["ocr_applied"] is True
    assert page["ocr_success"] is True
    assert page["extraction_method"] == "ocr"
    assert "Tesseract" in page["text"]
    assert page["ocr_confidence"] is not None
