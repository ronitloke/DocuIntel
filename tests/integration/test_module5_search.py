"""Real PostgreSQL/pgvector and full-text search verification for Module 5."""

from __future__ import annotations

import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from uuid import UUID

import fitz
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from app.core.config import Settings, get_settings
from app.db.session import Database, create_database
from app.main import create_app
from app.services.embeddings.sentence_transformer import EmbeddingService

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.module5,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_pdf(heading: str, body: str) -> bytes:
    """Create one native page with a heading and searchable body text."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 80), heading, fontsize=20)
    page.insert_textbox((72, 120, 520, 760), body, fontsize=12)
    try:
        return document.tobytes()
    finally:
        document.close()


def make_two_section_pdf() -> bytes:
    """Create two heading-scoped chunks for real semantic retrieval."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 80), "Employment Notice Policy", fontsize=20)
    page.insert_text(
        (72, 120),
        "Employees must give thirty days written notice before resignation.",
        fontsize=12,
    )
    page.insert_text((72, 220), "Product Catalog", fontsize=20)
    page.insert_text(
        (72, 260),
        "The company sells monitors and keyboards to business customers.",
        fontsize=12,
    )
    try:
        return document.tobytes()
    finally:
        document.close()


class FakeSearchEmbeddingModel:
    """Deterministic semantic buckets for fast endpoint and filter tests."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            lowered = value.lower()
            first = 1.0 if any(term in lowered for term in ("notice", "resignation", "employee")) else 0.0
            second = 1.0 if any(term in lowered for term in ("monitor", "keyboard", "product")) else 0.0
            third = 1.0 if "inv-2026-0043" in lowered else 0.0
            vectors.append([first, second, third] + [0.0] * 381)
        return vectors


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Migrate and clean the isolated Module 5 database."""

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


def client_for(
    database: Database,
    storage_directory: Path,
    embedding_service: EmbeddingService,
) -> TestClient:
    """Build an application bound to the isolated test database."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/search-384",
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


def upload_and_index(client: TestClient, filename: str, content: bytes) -> UUID:
    """Upload and index a fixture PDF, returning its persisted document ID."""

    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, BytesIO(content), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = UUID(upload.json()["document_id"])
    indexed = client.post(f"/api/v1/documents/{document_id}/index")
    assert indexed.status_code == 200, indexed.text
    return document_id


def test_migration_creates_generated_search_vector_and_gin_index(database: Database) -> None:
    """The new migration adds the PostgreSQL full-text representation and GIN index."""

    inspector = inspect(database.engine)
    columns = {column["name"]: column for column in inspector.get_columns("chunks")}
    assert "search_vector" in columns
    indexes = {index["name"]: index for index in inspector.get_indexes("chunks")}
    assert indexes["ix_chunks_search_vector_gin"]["dialect_options"]["postgresql_using"] == "gin"


def test_semantic_keyword_hybrid_filters_and_empty_results(
    database: Database,
    tmp_path: Path,
) -> None:
    """All modes rank persisted chunks, support SQL filters, and hide vectors."""

    embedder = EmbeddingService(
        Settings(embedding_model="test/search-384", embedding_dimension=384),
        model=FakeSearchEmbeddingModel(),
    )
    with client_for(database, tmp_path / "uploads", embedder) as client:
        employment_id = upload_and_index(
            client,
            "employment.pdf",
            make_pdf(
                "Employment Terms",
                "Employees must give thirty days written notice before resignation. "
                "Invoice reference INV-2026-0043.",
            ),
        )
        products_id = upload_and_index(
            client,
            "products.pdf",
            make_pdf("Product Catalog", "The company sells monitors and keyboards."),
        )

        semantic = client.post(
            "/api/v1/search",
            json={"query": "How much notice must an employee give?", "mode": "semantic", "top_k": 5},
        )
        assert semantic.status_code == 200, semantic.text
        assert UUID(semantic.json()["results"][0]["document_id"]) == employment_id
        assert semantic.json()["results"][0]["semantic_score"] is not None
        assert semantic.json()["results"][0]["keyword_score"] is None
        assert "embedding" not in semantic.json()["results"][0]

        keyword = client.post(
            "/api/v1/search",
            json={"query": "INV-2026-0043", "mode": "keyword", "top_k": 5},
        )
        assert keyword.status_code == 200, keyword.text
        assert UUID(keyword.json()["results"][0]["document_id"]) == employment_id
        assert keyword.json()["results"][0]["keyword_score"] is not None

        hybrid = client.post(
            "/api/v1/search",
            json={"query": "employee notice INV-2026-0043", "mode": "hybrid", "top_k": 5},
        )
        assert hybrid.status_code == 200, hybrid.text
        hybrid_results = hybrid.json()["results"]
        assert hybrid_results
        assert len({item["chunk_id"] for item in hybrid_results}) == len(hybrid_results)
        assert hybrid_results[0]["hybrid_score"] is not None
        assert hybrid_results[0]["retrieval_method"] == "hybrid"

        filtered = client.post(
            "/api/v1/search",
            json={
                "query": "monitors",
                "mode": "keyword",
                "top_k": 5,
                "filters": {"document_ids": [str(employment_id)]},
            },
        )
        assert filtered.status_code == 200
        assert filtered.json()["results"] == []

        content_filtered = client.post(
            "/api/v1/search",
            json={
                "query": "notice",
                "mode": "hybrid",
                "top_k": 5,
                "filters": {"content_types": ["table"]},
            },
        )
        assert content_filtered.status_code == 200
        assert content_filtered.json()["results"] == []

        no_results = client.post(
            "/api/v1/search",
            json={"query": "zzzz-nonexistent", "mode": "keyword", "top_k": 5},
        )
        assert no_results.status_code == 200
        assert no_results.json()["results"] == []
        assert no_results.json()["total_results"] == 0

        assert client.post("/api/v1/search", json={"query": "notice", "mode": "invalid"}).status_code == 422
        assert client.post("/api/v1/search", json={"query": "   "}).status_code == 422
        assert client.post("/api/v1/search", json={"query": "notice", "top_k": 51}).status_code == 422
        assert products_id != employment_id

    with database.engine.connect() as connection:
        assert connection.scalar(
            text(
                "SELECT count(*) FROM chunks c "
                "WHERE c.search_vector @@ websearch_to_tsquery('english', 'INV-2026-0043')"
            )
        ) >= 1


def test_real_all_minilm_semantic_search_against_pgvector(
    database: Database,
    tmp_path: Path,
) -> None:
    """The acceptance check uses the actual Module 4 model and SQL vector ranking."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    with client_for(
        database,
        tmp_path / "real-uploads",
        EmbeddingService(settings=settings),
    ) as client:
        document_id = upload_and_index(
            client,
            "real-search.pdf",
            make_two_section_pdf(),
        )
        response = client.post(
            "/api/v1/search",
            json={
                "query": "What resignation notice period is required?",
                "mode": "semantic",
                "top_k": 2,
                "filters": {"document_ids": [str(document_id)]},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()["results"][0]
        assert result["document_id"] == str(document_id)
        assert "notice" in result["text"].lower()
        assert result["semantic_score"] is not None
        assert result["start_page"] == 1
        assert result["end_page"] == 1
