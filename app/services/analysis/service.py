"""Document analysis orchestration over existing PostgreSQL chunks and Ollama."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import (
    AnalysisContentError,
    DatabaseNotConfiguredError,
    DocumentNotFoundError,
)
from app.db.models import Chunk, Document
from app.db.repository import DocumentRepository
from app.models.analysis import (
    AnalysisSource,
    DocumentClassificationRequest,
    DocumentClassificationResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
)
from app.services.analysis.classifier import DocumentClassifier
from app.services.analysis.summarizer import AnalysisChunk, DocumentSummarizer

logger = logging.getLogger(__name__)


class AnalysisService:
    """Coordinate bounded analysis without persisting generated results."""

    def __init__(
        self,
        repository: DocumentRepository | None,
        summarizer: DocumentSummarizer,
        classifier: DocumentClassifier,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.summarizer = summarizer
        self.classifier = classifier
        self.settings = settings

    async def summarize(
        self,
        document_id: UUID,
        request: DocumentSummaryRequest,
    ) -> DocumentSummaryResponse:
        """Load ordered chunks and generate a grounded hierarchical summary."""

        started = perf_counter()
        document, chunks, loading_time_ms = self._load_content(document_id)
        analysis_chunks = self._analysis_chunks(document, chunks)
        generation = await self.summarizer.summarize(
            analysis_chunks,
            style=request.style,
            batch_max_chars=self.settings.summary_batch_max_chars,
            final_max_chars=self.settings.summary_final_max_chars,
        )
        safe_summary = self.summarizer.enforce_final_grounding_boundary(
            generation.summary,
            analysis_chunks,
            style=request.style,
            source_ids=[f"S{index}" for index in range(1, len(analysis_chunks) + 1)],
            max_chars=self.settings.summary_final_max_chars,
        )
        total_time_ms = round((perf_counter() - started) * 1000, 3)
        logger.info(
            "Document summary completed document_id=%s chunks=%s style=%s total_ms=%.3f",
            document_id,
            len(chunks),
            request.style.value,
            total_time_ms,
        )
        return DocumentSummaryResponse(
            document_id=document.id,
            filename=document.original_filename,
            title=document.title,
            summary=safe_summary,
            style=request.style,
            model=self.summarizer.ollama_client.model,
            pages_represented=self._pages(chunks),
            chunks_represented=len(chunks),
            sources=self._sources(analysis_chunks),
            content_loading_time_ms=loading_time_ms,
            partial_generation_time_ms=generation.partial_generation_time_ms,
            final_synthesis_time_ms=generation.final_synthesis_time_ms,
            generation_time_ms=round(
                generation.partial_generation_time_ms + generation.final_synthesis_time_ms,
                3,
            ),
            total_time_ms=total_time_ms,
            grounding_verification_time_ms=generation.grounding_verification_time_ms,
            grounding_repair_time_ms=generation.grounding_repair_time_ms,
            grounding_verification_passes=generation.grounding_verification_passes,
        )

    async def classify(
        self,
        document_id: UUID,
        request: DocumentClassificationRequest,
    ) -> DocumentClassificationResponse:
        """Load ordered chunks and return a caller-label-constrained classification."""

        started = perf_counter()
        document, chunks, loading_time_ms = self._load_content(document_id)
        analysis_chunks = self._analysis_chunks(document, chunks)
        generation_started = perf_counter()
        result = await self.classifier.classify(
            analysis_chunks,
            labels=request.labels,
            context_max_chars=self.settings.summary_final_max_chars,
            batch_max_chars=self.settings.summary_batch_max_chars,
        )
        generation_time_ms = round((perf_counter() - generation_started) * 1000, 3)
        total_time_ms = round((perf_counter() - started) * 1000, 3)
        logger.info(
            "Document classification completed document_id=%s chunks=%s label=%s total_ms=%.3f",
            document_id,
            len(chunks),
            result.selected_label,
            total_time_ms,
        )
        return DocumentClassificationResponse(
            document_id=document.id,
            filename=document.original_filename,
            title=document.title,
            selected_label=result.selected_label,
            rationale=result.rationale,
            model=self.classifier.ollama_client.model,
            sources=self._sources(analysis_chunks),
            content_loading_time_ms=loading_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
        )

    def _load_content(self, document_id: UUID) -> tuple[Document, list[Chunk], float]:
        """Load one document and all ordered chunks in one repository operation."""

        if self.repository is None:
            raise DatabaseNotConfiguredError(
                "PostgreSQL is required for document analysis but is not configured."
            )
        started = perf_counter()
        document, chunks = self.repository.get_document_with_chunks(document_id)
        loading_time_ms = round((perf_counter() - started) * 1000, 3)
        if document is None:
            raise DocumentNotFoundError("The requested document was not found.")
        if not document.is_indexed:
            raise AnalysisContentError("The document must be indexed before analysis.")
        if not chunks:
            raise AnalysisContentError("The document has no indexed content to analyze.")
        return document, chunks, loading_time_ms

    @staticmethod
    def _analysis_chunks(document: Document, chunks: list[Chunk]) -> list[AnalysisChunk]:
        """Detach the minimum provenance needed by prompts and API responses."""

        return [
            AnalysisChunk(
                document_id=document.id,
                chunk_id=chunk.id,
                sequence_number=chunk.sequence_number,
                text=chunk.text,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_heading=chunk.section_heading,
                filename=document.original_filename,
            )
            for chunk in chunks
        ]

    @staticmethod
    def _pages(chunks: list[Chunk]) -> list[int]:
        """Return unique represented page numbers in ascending order."""

        pages: set[int] = set()
        for chunk in chunks:
            if chunk.start_page is None:
                continue
            end_page = chunk.end_page or chunk.start_page
            pages.update(range(chunk.start_page, end_page + 1))
        return sorted(pages)

    @staticmethod
    def _sources(chunks: list[AnalysisChunk]) -> list[AnalysisSource]:
        """Assign stable source labels in persisted chunk sequence order."""

        return [
            AnalysisSource(
                source_id=f"S{index}",
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                sequence_number=chunk.sequence_number,
                filename=chunk.filename,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_heading=chunk.section_heading,
                excerpt=chunk.text[:500],
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
