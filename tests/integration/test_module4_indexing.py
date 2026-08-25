"""Module 4 PostgreSQL, endpoint, and real local embedding verification."""

from __future__ import annotations

import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import fitz
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.db.session import Database, create_database
from app.main import create_app
from app.services.embeddings.sentence_transformer import EmbeddingService

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.module4,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_pdf(value: str) -> bytes:
    """Create a small native PDF that exercises heading and paragraph extraction."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 80), "Termination Conditions", fontsize=20)
    page.insert_text(
        (72, 120),
        value,
        fontsize=12,
    )
    page.insert_text(
        (72, 160),
        "Either party may terminate with thirty days written notice.",
        fontsize=12,
    )
    try:
        return document.tobytes()
    finally:
        document.close()


class FakeEmbeddingModel:
    """Deterministic 384-dimensional model for endpoint/idempotency tests."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        return [[0.01 * (index + 1)] * 384 for index, _ in enumerate(texts)]


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Migrate and clean the isolated Module 4 database."""

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
    embedding_service: EmbeddingService,
) -> TestClient:
    """Build a client bound to the isolated database and embedder."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/fake-384",
        embedding_dimension=384,
    )
    return TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=storage_directory,
            embedding_service=embedding_service,
        )
    )


def test_index_endpoint_persists_chunks_hides_vectors_and_is_idempotent(
    database: Database,
    tmp_path: Path,
) -> None:
    """Indexing replaces rows on re-run and public APIs never return raw vectors."""

    with application(
        database,
        tmp_path / "fake-uploads",
        EmbeddingService(
            Settings(embedding_model="test/fake-384", embedding_dimension=384),
            model=FakeEmbeddingModel(),
        ),
    ) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("module4-fake.pdf", BytesIO(make_pdf("Module 4 test content.")), "application/pdf")},
        )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["document_id"]

        indexed = client.post(f"/api/v1/documents/{document_id}/index")
        assert indexed.status_code == 200, indexed.text
        first = indexed.json()
        assert first["chunks_created"] >= 1
        assert first["embeddings_created"] == first["chunks_created"]
        assert first["embedding_dimension"] == 384

        chunks = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100")
        assert chunks.status_code == 200, chunks.text
        items = chunks.json()["items"]
        assert items
        assert "embedding" not in items[0]
        chunk_id = items[0]["id"]
        detail = client.get(f"/api/v1/documents/{document_id}/chunks/{chunk_id}")
        assert detail.status_code == 200
        assert "embedding" not in detail.json()

        indexed_again = client.post(f"/api/v1/documents/{document_id}/index")
        assert indexed_again.status_code == 200, indexed_again.text
        assert indexed_again.json()["chunks_created"] == first["chunks_created"]

    with database.engine.connect() as connection:
        assert connection.scalar(
            text("SELECT count(*) FROM chunks WHERE document_id = :document_id"),
            {"document_id": document_id},
        ) == first["chunks_created"]


def test_index_and_chunk_missing_resources_return_not_found(
    database: Database,
    tmp_path: Path,
) -> None:
    """Missing documents and chunks are scoped to the expected 404 behavior."""

    missing = "00000000-0000-0000-0000-000000000000"
    with application(
        database,
        tmp_path / "missing-uploads",
        EmbeddingService(
            Settings(embedding_model="test/fake-384", embedding_dimension=384),
            model=FakeEmbeddingModel(),
        ),
    ) as client:
        assert client.post(f"/api/v1/documents/{missing}/index").status_code == 404
        assert client.get(f"/api/v1/documents/{missing}/chunks").status_code == 404
        assert client.get(f"/api/v1/documents/{missing}/chunks/{missing}").status_code == 404


def test_actual_all_minilm_vectors_are_persisted(
    database: Database,
    tmp_path: Path,
) -> None:
    """The acceptance check uses the configured all-MiniLM-L6-v2 model and pgvector."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    with TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "real-uploads",
            embedding_service=EmbeddingService(settings=settings),
        )
    ) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("module4-real.pdf", BytesIO(make_pdf("Actual local embedding verification.")), "application/pdf")},
        )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["document_id"]
        indexed = client.post(f"/api/v1/documents/{document_id}/index")
        assert indexed.status_code == 200, indexed.text
        assert indexed.json()["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert indexed.json()["embedding_dimension"] == 384

        chunks = client.get(f"/api/v1/documents/{document_id}/chunks?page_size=100")
        assert chunks.status_code == 200
        assert chunks.json()["items"]

    with database.engine.connect() as connection:
        stored = connection.scalar(
            text(
                "SELECT count(*) FROM chunks "
                "WHERE document_id = :document_id AND embedding IS NOT NULL "
                "AND vector_dims(embedding) = 384"
            ),
            {"document_id": document_id},
        )
        assert stored == indexed.json()["chunks_created"]
