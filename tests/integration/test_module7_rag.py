"""PostgreSQL-backed and optional real-Ollama verification for Module 7."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from uuid import UUID

import fitz
import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.session import Database, create_database
from app.main import create_app
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.llm.ollama import OllamaClient
from app.services.reranking.cross_encoder import CrossEncoderReranker

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.module7,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_pdf() -> bytes:
    """Create two indexed sections with distinct retrieval subjects."""

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
    """Deterministic 384-dimensional vectors for database-backed RAG tests."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            lowered = value.lower()
            notice = 1.0 if any(term in lowered for term in ("notice", "resignation", "employee")) else 0.0
            product = 1.0 if any(term in lowered for term in ("monitor", "keyboard", "product")) else 0.0
            vectors.append([notice, product] + [0.0] * 382)
        return vectors


class FakeCrossEncoderModel:
    """Prefer the notice section while preserving the CrossEncoder contract."""

    device = "cpu"

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [4.0 if "notice" in candidate.lower() else 0.1 for _, candidate in pairs]


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Migrate and clean the isolated Module 7 database."""

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


def upload_and_index(client: TestClient, filename: str) -> UUID:
    """Upload and index the controlled PDF fixture."""

    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, BytesIO(make_pdf()), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document_id = UUID(upload.json()["document_id"])
    indexed = client.post(f"/api/v1/documents/{document_id}/index")
    assert indexed.status_code == 200, indexed.text
    return document_id


def test_rag_reuses_postgresql_search_reranking_and_filters(database: Database, tmp_path: Path) -> None:
    """The RAG endpoint receives only final filtered chunks from Modules 5–6."""

    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"response": "Employees must give thirty days written notice. [S1]"},
        )

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module7-embedding",
        embedding_dimension=384,
        ollama_model="test/module7-model",
        rag_default_top_k=5,
    )
    client = TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "uploads",
            embedding_service=EmbeddingService(settings, model=FakeEmbeddingModel()),
            reranker=CrossEncoderReranker(settings, model=FakeCrossEncoderModel()),
            ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler)),
        )
    )
    with client:
        document_id = upload_and_index(client, "module7-rag.pdf")
        response = client.post(
            "/api/v1/ask",
            json={
                "question": "What notice is required before resignation?",
                "top_k": 1,
                "search_mode": "hybrid",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].endswith("[S1]")
    assert body["citations"] == ["S1"]
    assert body["citations_valid"] is True
    assert body["model"] == "test/module7-model"
    assert body["sources"][0]["source_id"] == "S1"
    assert body["sources"][0]["document_id"] == str(document_id)
    assert body["sources"][0]["final_rank"] == 1
    assert body["sources"][0]["reranker_score"] == pytest.approx(4.0)
    payload = observed["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test/module7-model"
    assert payload["stream"] is False
    assert "thirty days" in payload["prompt"]
    assert "monitors" not in payload["prompt"]


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_RAG_TEST") != "1",
    reason="Set RUN_REAL_RAG_TEST=1 to verify the installed local Ollama model.",
)
def test_real_postgresql_rag_with_ollama(database: Database, tmp_path: Path) -> None:
    """Use actual indexed chunks, reranking, and the configured local Ollama model."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3.2:3b",
    )
    with TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "real-uploads",
            embedding_service=EmbeddingService(settings),
            reranker=CrossEncoderReranker(settings),
            ollama_client=OllamaClient(settings),
        )
    ) as client:
        document_id = upload_and_index(client, "module7-real-rag.pdf")
        response = client.post(
            "/api/v1/ask",
            json={
                "question": "What resignation notice period is required?",
                "top_k": 2,
                "search_mode": "hybrid",
                "rerank": True,
                "filters": {"document_ids": [str(document_id)]},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"]
    assert body["sources"]
    assert body["sources"][0]["document_id"] == str(document_id)
    assert body["citations_valid"] is True
    assert "thirty" in body["answer"].lower() or "30" in body["answer"]
