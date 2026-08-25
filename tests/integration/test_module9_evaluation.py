"""PostgreSQL-backed Module 9 retrieval and mocked-RAG evaluation tests."""

from __future__ import annotations

import json
import os
import asyncio
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
from app.db.repository import DocumentRepository
from app.db.session import Database, create_database
from app.evaluation.models import EvaluationCase, EvaluationConfiguration, EvaluationDataset
from app.evaluation.rag import RAGEvaluator
from app.evaluation.retrieval import RetrievalEvaluator
from app.main import create_app
from app.models.search import SearchMode, SearchFilters
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.llm.ollama import OllamaClient
from app.services.rag.service import RAGService
from app.services.reranking.cross_encoder import CrossEncoderReranker
from app.services.retrieval.search import SearchService

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.module9,
    pytest.mark.skipif(
        not TEST_DATABASE_URL,
        reason="Set TEST_DATABASE_URL to an isolated PostgreSQL/pgvector test database.",
    ),
]


def make_pdf() -> bytes:
    """Create a controlled policy/catalog document with a distractor section."""

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
    """Small deterministic vector model for evaluation integration tests."""

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        vectors: list[list[float]] = []
        for value in texts:
            lowered = value.lower()
            notice = float(any(term in lowered for term in ("notice", "resignation", "employee")))
            product = float(any(term in lowered for term in ("monitor", "keyboard", "product")))
            vectors.append([notice, product] + [0.0] * 382)
        return vectors


class FakeCrossEncoderModel:
    """Deterministic reranker that prefers notice evidence for the movement report."""

    device = "cpu"

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [4.0 if "notice" in candidate.lower() else 0.1 for _, candidate in pairs]


@pytest.fixture(scope="module")
def database() -> Iterator[Database]:
    """Apply the existing schema to the isolated evaluation database."""

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


def make_services(database: Database, ollama_transport: httpx.AsyncBaseTransport | None = None):
    """Build existing search/RAG services with deterministic test models."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="test/module9-embedding",
        embedding_dimension=384,
        ollama_model="test/module9-model",
    )
    embedding = EmbeddingService(settings, model=FakeEmbeddingModel())
    reranker = CrossEncoderReranker(settings, model=FakeCrossEncoderModel())
    search = SearchService(DocumentRepository(database), embedding, settings, reranker)
    rag = RAGService(search, OllamaClient(settings, transport=ollama_transport), settings)
    return settings, embedding, reranker, search, rag


def upload_and_index(
    database: Database,
    tmp_path: Path,
    *,
    filename: str = "module9-evaluation.pdf",
) -> UUID:
    """Create the controlled indexed document through the existing API."""

    settings, embedding, reranker, _, _ = make_services(database)
    with TestClient(
        create_app(
            settings=settings,
            database=database,
            storage_directory=tmp_path / "uploads",
            embedding_service=embedding,
            reranker=reranker,
        )
    ) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, BytesIO(make_pdf()), "application/pdf")},
        )
        assert response.status_code == 201, response.text
        document_id = UUID(response.json()["document_id"])
        indexed = client.post(f"/api/v1/documents/{document_id}/index")
        assert indexed.status_code == 200, indexed.text
        return document_id


def test_retrieval_evaluator_compares_modes_reranking_and_filters(
    database: Database,
    tmp_path: Path,
) -> None:
    """Evaluation calls SearchService and returns comparable measured reports."""

    document_id = upload_and_index(database, tmp_path)
    _, _, _, search, _ = make_services(database)
    dataset = EvaluationDataset(
        name="integration",
        cases=[
            EvaluationCase(
                id="notice",
                question="notice resignation",
                expected_document="module9-evaluation.pdf",
                filters=SearchFilters(document_ids=[document_id]),
            ),
            EvaluationCase(
                id="unsupported",
                question="What is the CEO favorite food?",
                expect_no_evidence=True,
                filters=SearchFilters(document_ids=[document_id]),
            ),
        ],
    )

    report = RetrievalEvaluator(search).compare(
        dataset,
        [
            EvaluationConfiguration(mode=SearchMode.SEMANTIC, top_k=3),
            EvaluationConfiguration(mode=SearchMode.KEYWORD, top_k=3),
            EvaluationConfiguration(mode=SearchMode.HYBRID, top_k=3),
            EvaluationConfiguration(mode=SearchMode.HYBRID, rerank=True, top_k=3),
        ],
    )

    assert [item.configuration.label for item in report.reports] == [
        "semantic",
        "keyword",
        "hybrid",
        "hybrid + rerank",
    ]
    assert all(item.summary.cases == 2 for item in report.reports)
    reranked_case = report.reports[-1].cases[0]
    assert reranked_case.base_rank is not None
    assert reranked_case.final_rank is not None
    assert reranked_case.rank_delta is not None
    assert report.reports[1].cases[1].no_evidence_correct is True


def test_rag_evaluator_uses_mocked_ollama_and_checks_no_evidence(
    database: Database,
    tmp_path: Path,
) -> None:
    """RAG evaluation remains deterministic and does not persist conversations."""

    filename = "module9-evaluation-rag.pdf"
    document_id = upload_and_index(database, tmp_path / "rag", filename=filename)
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"response": "Employees must give thirty days written notice. [S1]"})

    transport = httpx.MockTransport(handler)
    settings, embedding, reranker, _, _ = make_services(database, transport)
    search = SearchService(DocumentRepository(database), embedding, settings, reranker)
    rag = RAGService(search, OllamaClient(settings, transport=transport), settings)
    dataset = EvaluationDataset(
        name="rag-integration",
        cases=[
            EvaluationCase(
                id="notice",
                question="notice resignation",
                expected_document=filename,
                expected_facts=["thirty days", "written notice"],
                filters=SearchFilters(document_ids=[document_id]),
            ),
            EvaluationCase(
                id="unsupported",
                question="What is the CEO favorite food?",
                expect_no_evidence=True,
                filters=SearchFilters(document_ids=[document_id]),
            ),
        ],
    )

    import asyncio

    report = asyncio.run(
        RAGEvaluator(rag).evaluate(
            dataset,
            EvaluationConfiguration(mode=SearchMode.KEYWORD, rerank=False, top_k=1),
        )
    )
    assert report.summary.key_fact_coverage == 1.0
    # Citation quality is aggregated over cases where evidence was expected;
    # the no-evidence case is reported separately through its own metric.
    assert report.summary.citation_validity_rate == 1.0
    assert report.cases[0].expected_document_cited is True
    assert report.cases[0].evidence_support is True
    assert report.cases[1].no_evidence_correct is True
    assert len(calls) == 1


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODULE9_EVALUATION") != "1",
    reason="Set RUN_REAL_MODULE9_EVALUATION=1 for the local PostgreSQL/Ollama evaluation.",
)
def test_real_module9_evaluation_cold_and_warm(
    database: Database,
    tmp_path: Path,
) -> None:
    """Measure the complete evaluation layer against real local models and Ollama."""

    settings = Settings(
        database_url=TEST_DATABASE_URL,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3.2:3b",
        ollama_timeout_seconds=180,
    )
    filename = "module9-real-evaluation.pdf"
    document_id = upload_and_index(database, tmp_path / "real", filename=filename)
    embedding = EmbeddingService(settings)
    reranker = CrossEncoderReranker(settings)
    search = SearchService(DocumentRepository(database), embedding, settings, reranker)
    rag = RAGService(search, OllamaClient(settings), settings)
    dataset = EvaluationDataset(
        name="module9-real",
        cases=[
            EvaluationCase(
                id="notice",
                question="notice resignation",
                expected_document=filename,
                expected_facts=["thirty days", "written notice"],
                filters=SearchFilters(document_ids=[document_id]),
            )
        ],
    )

    retrieval_report = RetrievalEvaluator(search).compare(
        dataset,
        [
            EvaluationConfiguration(mode=SearchMode.SEMANTIC, top_k=2),
            EvaluationConfiguration(mode=SearchMode.KEYWORD, top_k=2),
            EvaluationConfiguration(mode=SearchMode.HYBRID, top_k=2),
            EvaluationConfiguration(mode=SearchMode.HYBRID, rerank=True, top_k=2),
        ],
    )
    cold = asyncio.run(
        RAGEvaluator(rag).evaluate(
            dataset,
            EvaluationConfiguration(mode=SearchMode.HYBRID, rerank=True, top_k=2),
        )
    )
    warm = asyncio.run(
        RAGEvaluator(rag).evaluate(
            dataset,
            EvaluationConfiguration(mode=SearchMode.HYBRID, rerank=True, top_k=2),
        )
    )

    cold_case = cold.cases[0]
    warm_case = warm.cases[0]
    assert cold_case.error is None, cold_case.error
    assert warm_case.error is None, warm_case.error
    assert cold_case.sources
    assert warm_case.sources
    assert cold_case.citations_valid is True
    assert warm_case.citations_valid is True
    assert cold_case.expected_document_cited is True
    assert warm_case.expected_document_cited is True
    print("MODULE9_REAL_RETRIEVAL=" + json.dumps(retrieval_report.model_dump(mode="json")))
    print("MODULE9_REAL_COLD=" + json.dumps(cold.model_dump(mode="json")))
    print("MODULE9_REAL_WARM=" + json.dumps(warm.model_dump(mode="json")))
