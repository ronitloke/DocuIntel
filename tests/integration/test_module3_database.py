"""Real PostgreSQL/pgvector verification for Module 3."""

from __future__ import annotations

import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import inspect, select, text

from app.core.config import Settings, get_settings
from app.db.repository import DocumentRepository
from app.db.session import Database, create_database
from app.core.exceptions import DocumentPersistenceError
from app.main import create_app
from app.models.documents import (
    DocumentIngestionResponse,
    DocumentStatus,
    PDFMetadata,
    PageExtraction,
)
from app.services.ocr.tesseract_ocr import OCRResult

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_pdf(text_value: str, with_table: bool = False) -> bytes:
    """Create a compact native PDF fixture."""

    document = fitz.open()
    page = document.new_page(width=400 if with_table else 595, height=300 if with_table else 842)
    page.insert_text((72, 80), text_value, fontsize=18)
    if with_table:
        x_positions = [50, 200, 300]
        y_positions = [110, 160, 210]
        for x_position in x_positions:
            page.draw_line((x_position, y_positions[0]), (x_position, y_positions[-1]))
        for y_position in y_positions:
            page.draw_line((x_positions[0], y_position), (x_positions[-1], y_position))
        page.insert_text((60, 140), "Product", fontsize=11)
        page.insert_text((210, 140), "Qty", fontsize=11)
        page.insert_text((60, 190), "Laptop", fontsize=11)
        page.insert_text((210, 190), "3", fontsize=11)
    try:
        return document.tobytes()
    finally:
        document.close()


def make_image_pdf() -> bytes:
    """Create an image-only PDF for persisted OCR-state verification."""

    source = fitz.open()
    source_page = source.new_page()
    source_page.insert_text((72, 130), "Persisted OCR text 12345", fontsize=22)
    pixmap = source_page.get_pixmap(dpi=150, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        output = BytesIO()
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
    finally:
        image.close()
        source.close()

    scanned = fitz.open()
    scanned_page = scanned.new_page(width=595, height=842)
    scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
    try:
        return scanned.tobytes()
    finally:
        scanned.close()


class FakeOCRService:
    """Deterministic OCR result for the database persistence test."""

    def is_available(self) -> bool:
        """Report an available OCR engine without invoking Tesseract."""

        return True

    def extract(self, image: Image.Image) -> OCRResult:
        """Return a successful result with a real-looking confidence value."""

        return OCRResult(
            text="Persisted OCR text 12345",
            success=True,
            confidence=88.5,
        )


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Migrate and clean an explicitly isolated PostgreSQL test database."""

    assert TEST_DATABASE_URL is not None
    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    get_settings.cache_clear()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, "head")
    database = create_database(Settings(database_url=TEST_DATABASE_URL))
    assert database is not None
    try:
        yield database
    finally:
        command.downgrade(config, "base")
        database.engine.dispose()
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url
        get_settings.cache_clear()


def application(
    database: Database,
    storage_directory: Path,
    ocr_service: FakeOCRService | None = None,
) -> TestClient:
    """Build a test client bound to the isolated database."""

    settings = Settings(database_url=TEST_DATABASE_URL)
    return TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=storage_directory,
            ocr_service=ocr_service,
        )
    )


def test_migration_and_pgvector_extension(database: Database) -> None:
    """The migration creates the extension and the nullable 384-dim vector column."""

    with database.engine.connect() as connection:
        extension = connection.scalar(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
    assert extension == "vector"
    inspector = inspect(database.engine)
    assert {"documents", "pages", "layout_elements", "document_tables", "chunks", "document_versions"}.issubset(
        set(inspector.get_table_names())
    )
    chunk_columns = {column["name"]: column for column in inspector.get_columns("chunks")}
    assert "embedding" in chunk_columns
    assert chunk_columns["embedding"]["nullable"] is True
    assert "384" in str(chunk_columns["embedding"]["type"])


def test_upload_persists_and_retrieves_structure(
    database: Database,
    tmp_path: Path,
) -> None:
    """Upload, list, detail, page, layout, and table retrieval use PostgreSQL rows."""

    content = make_pdf("Persisted native document", with_table=True)
    with application(database, tmp_path / "uploads") as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("persisted.pdf", BytesIO(content), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        document_id = response.json()["document_id"]

        listed = client.get("/api/v1/documents?page=1&page_size=10")
        assert listed.status_code == 200
        assert listed.json()["total"] == 1

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["database"] == "healthy"

        detail = client.get(f"/api/v1/documents/{document_id}")
        assert detail.status_code == 200
        assert detail.json()["summary"]["table_count"] == 1

        pages = client.get(f"/api/v1/documents/{document_id}/pages")
        assert pages.status_code == 200
        assert pages.json()["items"][0]["page_number"] == 1

        page = client.get(f"/api/v1/documents/{document_id}/pages/1")
        assert page.status_code == 200
        assert page.json()["layout_elements"]
        assert page.json()["tables"][0]["rows"] == [["Laptop", "3"]]

    with database.engine.connect() as connection:
        assert connection.scalar(select(text("count(*)")).select_from(text("documents"))) == 1
        assert connection.scalar(select(text("count(*)")).select_from(text("pages"))) == 1
        assert connection.scalar(select(text("count(*)")).select_from(text("layout_elements"))) >= 1
        assert connection.scalar(select(text("count(*)")).select_from(text("document_tables"))) == 1


def test_scanned_upload_persists_ocr_information(
    database: Database,
    tmp_path: Path,
) -> None:
    """OCR outcome and confidence survive the upload transaction."""

    with application(database, tmp_path / "uploads", FakeOCRService()) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("scanned.pdf", BytesIO(make_image_pdf()), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        document_id = response.json()["document_id"]
        page = client.get(f"/api/v1/documents/{document_id}/pages/1")

    assert page.status_code == 200
    assert page.json()["ocr_applied"] is True
    assert page.json()["ocr_success"] is True
    assert page.json()["ocr_confidence"] == 88.5


def test_duplicate_upload_returns_conflict_and_delete_removes_rows_and_file(
    database: Database,
    tmp_path: Path,
) -> None:
    """SHA-256 duplicate policy is 409, and deletion removes database/file state."""

    content = make_pdf("Duplicate policy document")
    storage = tmp_path / "uploads"
    with application(database, storage) as client:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("duplicate.pdf", BytesIO(content), "application/pdf")},
        )
        assert first.status_code == 201
        document_id = first.json()["document_id"]
        stored_filename = first.json()["stored_filename"]
        duplicate = client.post(
            "/api/v1/documents/upload",
            files={"file": ("renamed.pdf", BytesIO(content), "application/pdf")},
        )
        assert duplicate.status_code == 409
        assert (storage / stored_filename).is_file()
        deleted = client.delete(f"/api/v1/documents/{document_id}")
        assert deleted.status_code == 204
        assert not (storage / stored_filename).exists()
        assert client.get(f"/api/v1/documents/{document_id}").status_code == 404


def test_persistence_transaction_rolls_back_on_constraint_failure(database: Database) -> None:
    """A later invalid page cannot leave an earlier document/page partially committed."""

    response = DocumentIngestionResponse(
        document_id=uuid4(),
        original_filename="rollback.pdf",
        stored_filename=f"{uuid4()}.pdf",
        mime_type="application/pdf",
        file_size_bytes=10,
        checksum_sha256="a" * 64,
        page_count=2,
        pages_with_native_text=2,
        pages_requiring_ocr=0,
        status=DocumentStatus.READY,
        metadata=PDFMetadata(
            title=None,
            author=None,
            subject=None,
            keywords=None,
            creator=None,
            producer=None,
            creation_date=None,
            modification_date=None,
        ),
        pages=[
            PageExtraction(
                page_number=1,
                text="first",
                character_count=5,
                has_native_text=True,
                needs_ocr=False,
                extraction_method="native",
            ),
            PageExtraction(
                page_number=1,
                text="duplicate page number",
                character_count=21,
                has_native_text=True,
                needs_ocr=False,
                extraction_method="native",
            ),
        ],
    )
    repository = DocumentRepository(database)
    with pytest.raises(DocumentPersistenceError):
        repository.persist_ingestion(response)

    with database.engine.connect() as connection:
        document_id = str(response.document_id)
        assert connection.scalar(
            text("SELECT count(*) FROM documents WHERE id = :document_id"),
            {"document_id": document_id},
        ) == 0
        assert connection.scalar(
            text("SELECT count(*) FROM pages WHERE document_id = :document_id"),
            {"document_id": document_id},
        ) == 0
