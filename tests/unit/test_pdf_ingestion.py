"""Tests for Module 1 native PDF ingestion."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.main import create_app


def make_pdf(
    page_texts: list[str | None],
    metadata: dict[str, str] | None = None,
    encrypted: bool = False,
) -> bytes:
    """Create a small in-memory PDF fixture with PyMuPDF."""

    document = fitz.open()
    try:
        for page_text in page_texts:
            page = document.new_page()
            if page_text:
                page.insert_text((72, 72), page_text)
        if metadata:
            document.set_metadata(metadata)
        if encrypted:
            return document.tobytes(
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner-password",
                user_pw="user-password",
            )
        return document.tobytes()
    finally:
        document.close()


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Provide an API client with an isolated upload directory."""

    with TestClient(create_app(storage_directory=tmp_path / "uploads")) as test_client:
        yield test_client


def upload_pdf(
    client: TestClient,
    content: bytes,
    filename: str = "sample.pdf",
    content_type: str = "application/pdf",
) -> Response:
    """Submit a multipart PDF upload to the API."""

    return client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, BytesIO(content), content_type)},
    )


def test_valid_pdf_returns_metadata_text_and_uuid_storage(tmp_path: Path) -> None:
    """A valid PDF is stored under a generated UUID and extracted page by page."""

    storage_directory = tmp_path / "uploads"
    content = make_pdf(
        ["This page contains enough native PDF text for the default threshold."],
        metadata={"title": "Example", "author": "DocuIntel", "subject": "Testing"},
    )

    with TestClient(create_app(storage_directory=storage_directory)) as client:
        response = upload_pdf(client, content)

    assert response.status_code == 201
    payload = response.json()
    document_id = UUID(payload["document_id"])
    assert payload["original_filename"] == "sample.pdf"
    assert payload["stored_filename"] == f"{document_id}.pdf"
    assert not Path(payload["stored_filename"]).is_absolute()
    assert (storage_directory / payload["stored_filename"]).is_file()
    assert payload["file_size_bytes"] == len(content)
    assert payload["page_count"] == 1
    assert payload["pages_with_native_text"] == 1
    assert payload["pages_requiring_ocr"] == 0
    assert payload["metadata"]["title"] == "Example"
    assert payload["metadata"]["author"] == "DocuIntel"
    assert payload["metadata"]["subject"] == "Testing"
    assert payload["metadata"]["keywords"] is None
    assert payload["pages"][0]["page_number"] == 1
    assert "native PDF text" in payload["pages"][0]["text"]
    assert payload["pages"][0]["character_count"] > 0
    assert payload["pages"][0]["has_native_text"] is True
    assert payload["pages"][0]["needs_ocr"] is False
    assert payload["pages"][0]["extraction_method"] == "native"


def test_multi_page_pdf_preserves_page_order(client: TestClient) -> None:
    """Every page is returned in human-friendly one-based order."""

    response = upload_pdf(
        client,
        make_pdf(
            [
                "First page has enough text for native extraction.",
                "Second page has enough text for native extraction.",
                "Third page has enough text for native extraction.",
            ]
        ),
    )

    assert response.status_code == 201
    pages = response.json()["pages"]
    assert [page["page_number"] for page in pages] == [1, 2, 3]
    assert ["page" in page["text"] for page in pages] == [True, True, True]


def test_blank_page_is_flagged_for_future_ocr_but_not_processed(client: TestClient) -> None:
    """A blank page is an OCR candidate and remains marked as native extraction."""

    response = upload_pdf(
        client,
        make_pdf(["This page has native text above the OCR threshold.", None]),
    )

    assert response.status_code == 201
    payload = response.json()
    blank_page = payload["pages"][1]
    assert payload["pages_with_native_text"] == 1
    assert payload["pages_requiring_ocr"] == 1
    assert blank_page == {
        "page_number": 2,
        "text": "",
        "character_count": 0,
        "has_native_text": False,
        "needs_ocr": True,
        "extraction_method": "native",
    }


def test_unicode_and_traversal_like_filename_is_metadata_only(client: TestClient) -> None:
    """Unsafe-looking names cannot affect the UUID-based physical path."""

    original_filename = r"..\..\résumé.pdf"
    response = upload_pdf(client, make_pdf(["A valid page with enough native text."]), original_filename)

    assert response.status_code == 201
    payload = response.json()
    assert payload["original_filename"] == original_filename
    assert payload["stored_filename"].endswith(".pdf")
    assert ".." not in payload["stored_filename"]
    assert "\\" not in payload["stored_filename"]


def test_invalid_extension_is_rejected(client: TestClient) -> None:
    """Only .pdf filenames are accepted."""

    response = upload_pdf(client, make_pdf(["Valid PDF content."]), "document.txt")

    assert response.status_code == 415
    assert ".pdf extension" in response.json()["detail"]


def test_unexpected_content_type_is_rejected(client: TestClient) -> None:
    """A valid signature cannot override an unsupported declared MIME type."""

    response = upload_pdf(
        client,
        make_pdf(["Valid PDF content."]),
        content_type="text/plain",
    )

    assert response.status_code == 415
    assert "content type" in response.json()["detail"]


def test_fake_pdf_signature_is_rejected(client: TestClient) -> None:
    """A non-PDF renamed with a .pdf extension is rejected."""

    response = upload_pdf(client, b"This is not a PDF.")

    assert response.status_code == 415
    assert "signature" in response.json()["detail"]


def test_empty_upload_is_rejected(client: TestClient) -> None:
    """Zero-byte uploads are rejected before PyMuPDF is called."""

    response = upload_pdf(client, b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_corrupted_pdf_is_rejected_cleanly(client: TestClient) -> None:
    """A PDF signature alone is insufficient if PyMuPDF cannot parse the file."""

    response = upload_pdf(client, b"%PDF-1.7\nnot a complete PDF document")

    assert response.status_code == 422
    assert "parsed as a PDF" in response.json()["detail"]


def test_failed_ingestion_leaves_no_partial_upload(tmp_path: Path) -> None:
    """Rejected files do not remain in the upload directory."""

    storage_directory = tmp_path / "uploads"
    with TestClient(create_app(storage_directory=storage_directory)) as client:
        response = upload_pdf(client, b"%PDF-1.7\nnot a complete PDF document")

    assert response.status_code == 422
    assert list(storage_directory.glob("*")) == []


def test_oversized_upload_uses_configured_limit(tmp_path: Path) -> None:
    """The configured limit is enforced while streaming the upload."""

    settings = Settings(max_upload_size_mb=1)
    content = b"%PDF-1.7\n" + b"x" * (1024 * 1024)

    with TestClient(
        create_app(settings=settings, storage_directory=tmp_path / "uploads")
    ) as client:
        response = upload_pdf(client, content)

    assert response.status_code == 413
    assert "1 MB upload limit" in response.json()["detail"]


def test_encrypted_pdf_is_rejected_without_unlocking(client: TestClient) -> None:
    """Password-protected PDFs are reported as unsupported in Module 1."""

    response = upload_pdf(
        client,
        make_pdf(["Secret page content."], encrypted=True),
    )

    assert response.status_code == 422
    assert "Password-protected" in response.json()["detail"]


def test_upload_endpoint_is_documented_as_multipart(client: TestClient) -> None:
    """OpenAPI exposes a file-upload request body for Swagger UI."""

    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"]["/api/v1/documents/upload"]["post"]

    assert operation["requestBody"]["content"]["multipart/form-data"]
    assert operation["responses"]["201"]["content"]["application/json"]
