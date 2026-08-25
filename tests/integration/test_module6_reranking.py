"""PostgreSQL-backed API and real-model verification for Module 6."""

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

from app.core.config import Settings, get_settings
from app.db.session import Database, create_database
from app.main import create_app
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.reranking.cross_encoder import CrossEncoderReranker

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.module6,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_two_section_pdf() -> bytes:
    """Create one PDF with distinct notice and product chunks."""

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


class FakeEmbeddingModel:
    """Deterministic 384-dimensional vectors for fast database tests."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            lowered = value.lower()
            notice = 1.0 if any(term in lowered for term in ("notice", "resignation", "employee")) else 0.0
            product = 1.0 if any(term in lowered for term in ("monitor", "keyboard", "product")) else 0.0
            vectors.append([notice, product] + [0.0] * 382)
        return vectors


class FakeCrossEncoderModel:
    """Controlled scorer that demonstrates content-driven rank movement."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        self.calls.append(pairs)
        return [5.4 if "notice" in candidate.lower() else 0.1 for _, candidate in pairs]


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Apply all migrations to an isolated database for this module."""

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
    *,
    settings: Settings | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> TestClient:
    """Build an application using the isolated database and supplied services."""

    app_settings = settings or Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module6-embedding",
        embedding_dimension=384,
    )
    return TestClient(
        create_app(
            settings=app_settings,
            database=database,
            storage_directory=storage_directory,
            embedding_service=embedding_service,
            reranker=reranker,
        )
    )


def upload_and_index(client: TestClient, filename: str) -> UUID:
    """Upload and index the controlled PDF fixture."""

    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, BytesIO(make_two_section_pdf()), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = UUID(upload.json()["document_id"])
    indexed = client.post(f"/api/v1/documents/{document_id}/index")
    assert indexed.status_code == 200, indexed.text
    return document_id


def fake_client(database: Database, storage_directory: Path) -> tuple[TestClient, FakeCrossEncoderModel]:
    """Create a test client with deterministic embedding and reranking models."""

    model = FakeCrossEncoderModel()
    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module6-embedding",
        embedding_dimension=384,
        rerank_candidate_count=20,
        rerank_candidate_multiplier=4,
    )
    client = client_for(
        database,
        storage_directory,
        EmbeddingService(settings, model=FakeEmbeddingModel()),
        settings=settings,
        reranker=CrossEncoderReranker(settings, model=model),
    )
    return client, model


def test_rerank_false_preserves_base_search_contract(database: Database, tmp_path: Path) -> None:
    """The opt-in flag leaves Module 5 result ordering and scores intact."""

    client, _ = fake_client(database, tmp_path / "base")
    with client:
        document_id = upload_and_index(client, "module6-base.pdf")
        response = client.post(
            "/api/v1/search",
            json={
                "query": "employee notice",
                "mode": "hybrid",
                "top_k": 2,
                "rerank": False,
                "filters": {"document_ids": [str(document_id)]},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reranked"] is False
    assert body["rerank_time_ms"] is None
    assert body["retrieval_time_ms"] >= 0
    assert body["total_search_time_ms"] == body["search_time_ms"]
    assert body["results"]
    assert all(item["reranked"] is False and item["base_rank"] is None for item in body["results"])


@pytest.mark.parametrize("mode", ["semantic", "keyword", "hybrid"])
def test_reranking_supports_all_search_modes_and_filters(
    database: Database,
    tmp_path: Path,
    mode: str,
) -> None:
    """Semantic, keyword, and hybrid retrieval all share one reranking stage."""

    client, model = fake_client(database, tmp_path / mode)
    with client:
        document_id = upload_and_index(client, f"module6-{mode}.pdf")
        query = "employee notice" if mode != "keyword" else "notice"
        response = client.post(
            "/api/v1/search",
            json={
                "query": query,
                "mode": mode,
                "top_k": 1,
                "rerank": True,
                "filters": {"document_ids": [str(document_id)], "page_start": 1, "page_end": 1},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reranked"] is True
    assert body["rerank_time_ms"] is not None
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["reranked"] is True
    assert result["rank"] == 1
    assert result["base_rank"] >= 1
    assert result["rerank_score"] == pytest.approx(5.4)
    assert result["document_id"] == str(document_id)
    assert result["start_page"] == 1
    assert "notice" in result["text"].lower()
    assert model.calls


def test_missing_reranker_model_returns_controlled_service_error(
    database: Database,
    tmp_path: Path,
) -> None:
    """A requested but unavailable model returns 503 instead of base results."""

    settings = Settings(database_url=TEST_DATABASE_URL, embedding_dimension=384)

    def failing_loader(*_: object) -> object:
        raise RuntimeError("not cached")

    reranker = CrossEncoderReranker(settings, model_loader=failing_loader)
    with client_for(
        database,
        tmp_path / "failure",
        EmbeddingService(settings, model=FakeEmbeddingModel()),
        settings=settings,
        reranker=reranker,
    ) as client:
        document_id = upload_and_index(client, "module6-failure.pdf")
        response = client.post(
            "/api/v1/search",
            json={
                "query": "notice",
                "mode": "semantic",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )

    assert response.status_code == 503
    assert "reranker" in response.json()["detail"].lower()


def test_real_cross_encoder_reranks_postgresql_chunks(database: Database, tmp_path: Path) -> None:
    """The configured MiniLM CrossEncoder scores actual indexed PostgreSQL chunks."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    with client_for(
        database,
        tmp_path / "real",
        EmbeddingService(settings),
        settings=settings,
        reranker=CrossEncoderReranker(settings),
    ) as client:
        document_id = upload_and_index(client, "module6-real.pdf")
        response = client.post(
            "/api/v1/search",
            json={
                "query": "What resignation notice period is required?",
                "mode": "hybrid",
                "top_k": 2,
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"]
    assert len(body["results"]) <= 2
    assert body["rerank_time_ms"] is not None
    assert all(item["reranked"] for item in body["results"])
    assert all(item["rerank_score"] is not None for item in body["results"])
    assert body["results"][0]["document_id"] == str(document_id)
    assert "notice" in body["results"][0]["text"].lower()
    scores = [item["rerank_score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)
