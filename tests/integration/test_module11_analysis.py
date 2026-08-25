"""PostgreSQL-backed API verification for Module 11 analysis."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

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

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


class FakeEmbeddingModel:
    """Deterministic embedding fixture that satisfies the 384-dimensional contract."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        return [[1.0 if "notice" in text.lower() else 0.0] + [0.0] * 383 for text in texts]


def make_pdf() -> bytes:
    """Create one small employment-policy PDF for analysis acceptance coverage."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 80), "Employment Notice Policy", fontsize=20)
    page.insert_text(
        (72, 120),
        "Employees must provide thirty days of written notice before resignation.",
        fontsize=12,
    )
    try:
        return document.tobytes()
    finally:
        document.close()


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Migrate and clean the isolated Module 11 database."""

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


def test_summary_and_classification_api_use_ordered_postgres_chunks(
    database: Database,
    tmp_path: Path,
) -> None:
    """Both endpoints analyze persisted indexed content and return provenance."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("format") == "json":
            prompt = str(payload.get("prompt", ""))
            if "allowed_labels" not in prompt:
                return httpx.Response(
                    200,
                    json={
                        "response": json.dumps(
                            {
                                "claims": [
                                    {
                                        "claim": "The document states a written resignation notice period.",
                                        "supported": True,
                                        "source_labels": ["S1"],
                                        "supporting_evidence": (
                                            "Employees must provide thirty days of written notice "
                                            "before resignation."
                                        ),
                                        "reason": "Directly supported by the indexed chunk.",
                                    }
                                ],
                                "has_unsupported_claims": False,
                                "repaired_summary": "",
                            }
                        )
                    },
                )
            return httpx.Response(
                200,
                json={
                    "response": json.dumps(
                        {
                            "selected_label": "Employment Policy",
                            "rationale": "The document states a written resignation notice period.",
                        }
                    )
                },
            )
        return httpx.Response(
            200,
            json={"response": "The document states a thirty-day written resignation notice. [S1]"},
        )

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module11-embedding",
        ollama_model="test/module11-model",
        summary_batch_max_chars=3000,
        summary_final_max_chars=6000,
    )
    app = create_app(
        settings=settings,
        database=database,
        storage_directory=tmp_path / "uploads",
        embedding_service=EmbeddingService(settings, model=FakeEmbeddingModel()),
        ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("employment.pdf", BytesIO(make_pdf()), "application/pdf")},
        )
        assert upload.status_code == 201, upload.text
        document_id = UUID(upload.json()["document_id"])
        indexed = client.post(f"/api/v1/documents/{document_id}/index")
        assert indexed.status_code == 200, indexed.text

        summary = client.post(
            f"/api/v1/documents/{document_id}/summary",
            json={"style": "brief"},
        )
        classification = client.post(
            f"/api/v1/documents/{document_id}/classify",
            json={
                "labels": [
                    "Employment Policy",
                    "Expense Policy",
                    "Data Retention Policy",
                    "Office Operations",
                    "Other",
                ]
            },
        )

    assert summary.status_code == 200, summary.text
    assert summary.json()["chunks_represented"] == 1
    assert summary.json()["sources"][0]["source_id"] == "S1"
    assert "thirty days of written notice before resignation. [S1]" in summary.json()["summary"]
    assert summary.json()["grounding_verification_passes"] >= 1
    assert summary.json()["grounding_verification_time_ms"] >= 0
    assert summary.json()["grounding_repair_time_ms"] >= 0
    assert classification.status_code == 200, classification.text
    assert classification.json()["selected_label"] == "Employment Policy"
    assert classification.json()["sources"][0]["start_page"] == 1


def test_analysis_missing_document_and_invalid_labels_are_controlled(
    database: Database,
    tmp_path: Path,
) -> None:
    """Missing content and request validation never call a real provider."""

    settings = Settings(database_url=TEST_DATABASE_URL, ollama_model="test/module11-model")
    app = create_app(settings=settings, database=database, storage_directory=tmp_path / "uploads")
    missing_id = uuid4()
    with TestClient(app) as client:
        missing = client.post(
            f"/api/v1/documents/{missing_id}/summary",
            json={"style": "brief"},
        )
        invalid = client.post(
            f"/api/v1/documents/{missing_id}/classify",
            json={"labels": ["Only one"]},
        )

    assert missing.status_code == 404
    assert invalid.status_code == 422
