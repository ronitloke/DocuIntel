"""Document indexing orchestration for chunk generation and vector persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from app.core.exceptions import DocumentIngestionError, DocumentNotFoundError
from app.db.repository import DocumentRepository
from app.services.chunking.structure_aware import StructureAwareChunker
from app.services.embeddings.sentence_transformer import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Safe indexing summary returned by the API layer."""

    document_id: UUID
    chunks_created: int
    embeddings_created: int
    embedding_model: str
    embedding_dimension: int
    status: str = "indexed"


class DocumentIndexingService:
    """Generate all vectors before one transactional replace of a document index."""

    def __init__(
        self,
        repository: DocumentRepository | None,
        chunker: StructureAwareChunker,
        embedding_service: EmbeddingService,
    ) -> None:
        self.repository = repository
        self.chunker = chunker
        self.embedding_service = embedding_service

    def index_document(self, document_id: UUID) -> IndexingResult:
        """Index one existing document without disturbing a valid prior index on failure."""

        repository = self._require_repository()
        started = perf_counter()
        logger.info("Document indexing started document_id=%s", document_id)
        document = repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError("The requested document was not found.")

        chunk_started = perf_counter()
        drafts = self.chunker.build_chunks(document)
        logger.info(
            "Document chunk generation completed document_id=%s chunks=%s duration_seconds=%.3f",
            document_id,
            len(drafts),
            perf_counter() - chunk_started,
        )

        embedding_started = perf_counter()
        embeddings = self.embedding_service.embed_texts([draft.text for draft in drafts]) if drafts else []
        logger.info(
            "Document embedding generation completed document_id=%s embeddings=%s duration_seconds=%.3f",
            document_id,
            len(embeddings),
            perf_counter() - embedding_started,
        )

        persistence_started = perf_counter()
        try:
            repository.replace_document_index(
                document_id=document_id,
                drafts=drafts,
                embeddings=embeddings,
                embedding_model=self.embedding_service.model_name,
                embedding_dimension=self.embedding_service.dimension,
            )
        except DocumentIngestionError:
            logger.exception("Document indexing persistence failed document_id=%s", document_id)
            raise
        finally:
            logger.info(
                "Document indexing persistence completed document_id=%s duration_seconds=%.3f",
                document_id,
                perf_counter() - persistence_started,
            )

        logger.info(
            "Document indexing completed document_id=%s chunks=%s duration_seconds=%.3f",
            document_id,
            len(drafts),
            perf_counter() - started,
        )
        return IndexingResult(
            document_id=document_id,
            chunks_created=len(drafts),
            embeddings_created=len(embeddings),
            embedding_model=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.dimension,
        )

    def _require_repository(self) -> DocumentRepository:
        """Require configured persistence before starting the workflow."""

        if self.repository is None:
            from app.core.exceptions import DatabaseNotConfiguredError

            raise DatabaseNotConfiguredError(
                "PostgreSQL is required for document indexing but is not configured."
            )
        return self.repository
