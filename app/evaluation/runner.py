"""Runtime construction and top-level Module 9 evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.db.repository import DocumentRepository
from app.db.session import Database, create_database
from app.evaluation.models import EvaluationConfiguration
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.llm.ollama import OllamaClient
from app.services.rag.service import RAGService
from app.services.reranking.cross_encoder import CrossEncoderReranker
from app.services.retrieval.search import SearchService


@dataclass(slots=True)
class EvaluationRuntime:
    """Read-only services used by evaluation commands."""

    database: Database
    search_service: SearchService
    rag_service: RAGService

    def close(self) -> None:
        """Release the SQLAlchemy engine without changing database data."""

        self.database.engine.dispose()


def build_runtime(settings: Settings) -> EvaluationRuntime:
    """Construct existing runtime services without running migrations or writing data."""

    database = create_database(settings)
    if database is None:
        raise RuntimeError("DATABASE_URL or PostgreSQL settings are required for evaluation.")
    embedding_service = EmbeddingService(settings=settings)
    reranker = CrossEncoderReranker(settings=settings)
    search_service = SearchService(
        repository=DocumentRepository(database),
        embedding_service=embedding_service,
        settings=settings,
        reranker=reranker,
    )
    rag_service = RAGService(
        search_service=search_service,
        ollama_client=OllamaClient(settings=settings),
        settings=settings,
    )
    return EvaluationRuntime(database, search_service, rag_service)


def default_configuration(mode: str, *, rerank: bool, top_k: int) -> EvaluationConfiguration:
    """Create a validated configuration from CLI values."""

    return EvaluationConfiguration(mode=mode, rerank=rerank, top_k=top_k)
