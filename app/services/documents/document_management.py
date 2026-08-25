"""Document persistence orchestration and API-facing projections."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from app.core.exceptions import (
    DatabaseNotConfiguredError,
    DocumentIngestionError,
    DocumentNotFoundError,
    DocumentPersistenceError,
)
from app.db.models import Chunk, Document, Page
from app.db.repository import DocumentRepository
from app.models.documents import (
    DocumentDetailResponse,
    ChunkListItem,
    ChunkListResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentStatus,
    DocumentSummary,
    ExtractedTable,
    LayoutElement,
    PageListItem,
    PageListResponse,
    PersistedPageResponse,
    DocumentIngestionResponse,
    PDFMetadata,
)
from app.services.documents.pdf_ingestion import PDFIngestionService, UploadStream

logger = logging.getLogger(__name__)


class DocumentManagementService:
    """Coordinate PDF extraction, database transactions, and document APIs."""

    def __init__(
        self,
        ingestion_service: PDFIngestionService,
        repository: DocumentRepository | None,
        storage_directory: Path,
        persistence_required: bool,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.repository = repository
        self.storage_directory = storage_directory
        self.persistence_required = persistence_required

    async def ingest(self, upload: UploadStream) -> DocumentIngestionResponse:
        """Extract a PDF and commit every extracted record atomically."""

        response = await self.ingestion_service.ingest(upload)
        if self.repository is None:
            if self.persistence_required:
                self._remove_stored_file(response.stored_filename)
                raise DatabaseNotConfiguredError(
                    "PostgreSQL is required for document uploads but is not configured."
                )
            logger.warning(
                "Document persistence is disabled for an isolated application instance "
                "document_id=%s",
                response.document_id,
            )
            return response

        try:
            self.repository.persist_ingestion(response)
        except DocumentIngestionError:
            self._remove_stored_file(response.stored_filename)
            raise
        except Exception as exc:
            self._remove_stored_file(response.stored_filename)
            raise DocumentPersistenceError(
                "The document could not be persisted safely."
            ) from exc
        return response

    def list_documents(
        self,
        page: int,
        page_size: int,
        status: DocumentStatus | None,
    ) -> DocumentListResponse:
        """Return a paginated document projection."""

        repository = self._require_repository()
        documents, total = repository.list_documents(page, page_size, status)
        return DocumentListResponse(
            items=[self._to_list_item(document) for document in documents],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._total_pages(total, page_size),
        )

    def get_document(self, document_id: UUID) -> DocumentDetailResponse:
        """Return document metadata and processing counts."""

        document = self._require_repository().get_document(document_id)
        if document is None:
            raise DocumentNotFoundError("The requested document was not found.")
        return self._to_detail(document)

    def list_pages(
        self,
        document_id: UUID,
        page: int,
        page_size: int,
    ) -> PageListResponse:
        """Return a paginated page metadata projection."""

        repository = self._require_repository()
        if not repository.document_exists(document_id):
            raise DocumentNotFoundError("The requested document was not found.")
        pages, total = repository.list_pages(document_id, page, page_size)
        return PageListResponse(
            items=[self._to_page_list_item(item) for item in pages],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._total_pages(total, page_size),
        )

    def get_page(self, document_id: UUID, page_number: int) -> PersistedPageResponse:
        """Return one page's text, OCR state, layout, and tables."""

        page = self._require_repository().get_page(document_id, page_number)
        if page is None:
            raise DocumentNotFoundError("The requested page was not found.")
        return self._to_page_detail(page)

    def list_chunks(
        self,
        document_id: UUID,
        page: int,
        page_size: int,
    ) -> ChunkListResponse:
        """Return public chunk projections for one persisted document."""

        repository = self._require_repository()
        if not repository.document_exists(document_id):
            raise DocumentNotFoundError("The requested document was not found.")
        chunks, total = repository.list_chunks(document_id, page, page_size)
        return ChunkListResponse(
            items=[self._to_chunk_item(chunk) for chunk in chunks],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=self._total_pages(total, page_size),
        )

    def get_chunk(self, document_id: UUID, chunk_id: UUID) -> ChunkListItem:
        """Return one public chunk projection scoped to its document."""

        chunk = self._require_repository().get_chunk(document_id, chunk_id)
        if chunk is None:
            raise DocumentNotFoundError("The requested chunk was not found.")
        return self._to_chunk_item(chunk)

    def delete_document(self, document_id: UUID) -> None:
        """Delete database dependents and then the associated safe local PDF."""

        stored_filename = self._require_repository().delete_document(document_id)
        if stored_filename is None:
            raise DocumentNotFoundError("The requested document was not found.")
        self._remove_stored_file(stored_filename, raise_on_error=True)

    def _require_repository(self) -> DocumentRepository:
        """Return the repository or raise a controlled readiness error."""

        if self.repository is None:
            raise DatabaseNotConfiguredError(
                "PostgreSQL is required for document management but is not configured."
            )
        return self.repository

    def _remove_stored_file(
        self,
        stored_filename: str,
        raise_on_error: bool = False,
    ) -> None:
        """Remove only a generated filename inside the configured upload root."""

        root = self.storage_directory.resolve()
        candidate = (root / stored_filename).resolve()
        if candidate.parent != root or candidate.name != stored_filename:
            logger.error("Refusing unsafe stored filename cleanup filename=%s", stored_filename)
            if raise_on_error:
                raise DocumentPersistenceError("The stored PDF path is unsafe.")
            return
        try:
            candidate.unlink(missing_ok=True)
        except OSError as exc:
            logger.exception("Stored PDF cleanup failed filename=%s", stored_filename)
            if raise_on_error:
                raise DocumentPersistenceError(
                    "The document metadata was deleted but its stored PDF could not be removed."
                ) from exc

    @staticmethod
    def _total_pages(total: int, page_size: int) -> int:
        """Return a ceiling page count without loading all rows."""

        return (total + page_size - 1) // page_size if total else 0

    @staticmethod
    def _to_list_item(document: Document) -> DocumentListItem:
        """Project an ORM document into a compact public schema."""

        return DocumentListItem(
            id=document.id,
            original_filename=document.original_filename,
            stored_filename=document.stored_filename,
            mime_type=document.mime_type,
            file_size_bytes=document.file_size_bytes,
            page_count=document.page_count,
            status=document.status,
            checksum_sha256=document.checksum_sha256,
            created_at=document.created_at,
            updated_at=document.updated_at,
            is_indexed=document.is_indexed,
            indexed_at=document.indexed_at,
            chunk_count=document.chunk_count,
            embedding_model=document.embedding_model,
            embedding_dimension=document.embedding_dimension,
        )

    @classmethod
    def _to_detail(cls, document: Document) -> DocumentDetailResponse:
        """Project a loaded document into metadata plus aggregate statistics."""

        pages = document.pages
        layout_count = sum(len(page.layout_elements) for page in pages)
        table_count = sum(len(page.tables) for page in pages)
        summary = DocumentSummary(
            page_count=document.page_count,
            pages_with_native_text=sum(page.has_native_text for page in pages),
            pages_requiring_ocr=sum(page.needs_ocr for page in pages),
            pages_processed_by_ocr=sum(page.ocr_applied for page in pages),
            ocr_failed_pages=sum(
                page.ocr_applied and page.ocr_success is False for page in pages
            ),
            unresolved_ocr_pages=sum(page.needs_ocr for page in pages),
            heading_count=sum(
                element.element_type == "heading"
                for page in pages
                for element in page.layout_elements
            ),
            table_count=table_count,
            layout_element_count=layout_count,
        )
        item = cls._to_list_item(document)
        return DocumentDetailResponse(
            **item.model_dump(),
            metadata=cls._metadata(document),
            summary=summary,
        )

    @staticmethod
    def _metadata(document: Document) -> PDFMetadata:
        """Convert persisted PDF metadata without inventing values."""

        return PDFMetadata(
            title=document.title,
            author=document.author,
            subject=document.subject,
            keywords=document.keywords,
            creator=document.creator,
            producer=document.producer,
            creation_date=document.creation_date,
            modification_date=document.modification_date,
        )

    @staticmethod
    def _to_page_list_item(page: Page) -> PageListItem:
        """Project one page without returning its potentially large text."""

        return PageListItem(
            id=page.id,
            document_id=page.document_id,
            page_number=page.page_number,
            character_count=page.character_count,
            has_native_text=page.has_native_text,
            needs_ocr=page.needs_ocr,
            extraction_method=page.extraction_method,
            ocr_applied=page.ocr_applied,
            ocr_success=page.ocr_success,
            ocr_confidence=page.ocr_confidence,
            created_at=page.created_at,
        )

    @staticmethod
    def _to_page_detail(page: Page) -> PersistedPageResponse:
        """Project one loaded page and its child structure rows."""

        return PersistedPageResponse(
            id=page.id,
            document_id=page.document_id,
            page_number=page.page_number,
            text=page.extracted_text,
            character_count=page.character_count,
            has_native_text=page.has_native_text,
            needs_ocr=page.needs_ocr,
            extraction_method=page.extraction_method,
            ocr_applied=page.ocr_applied,
            ocr_success=page.ocr_success,
            ocr_confidence=page.ocr_confidence,
            created_at=page.created_at,
            layout_elements=[
                LayoutElement(
                    element_type=element.element_type,
                    text=element.text,
                    bbox=[element.bbox_x0, element.bbox_y0, element.bbox_x1, element.bbox_y1],
                    font_size=element.font_size,
                    is_bold=element.is_bold,
                )
                for element in page.layout_elements
            ],
            tables=[
                ExtractedTable(
                    table_index=table.table_index,
                    page_number=page.page_number,
                    bbox=[table.bbox_x0, table.bbox_y0, table.bbox_x1, table.bbox_y1],
                    headers=table.headers,
                    rows=table.rows,
                )
                for table in page.tables
            ],
        )

    @staticmethod
    def _to_chunk_item(chunk: Chunk) -> ChunkListItem:
        """Project a chunk while omitting its raw pgvector embedding."""

        return ChunkListItem(
            id=chunk.id,
            document_id=chunk.document_id,
            sequence_number=chunk.sequence_number,
            text=chunk.text,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            section_heading=chunk.section_heading,
            content_type=chunk.content_type,
            contains_ocr=chunk.contains_ocr,
            character_count=chunk.character_count,
            token_count=chunk.token_count,
            embedding_model=chunk.embedding_model,
            embedding_dimension=chunk.embedding_dimension,
            fingerprint_sha256=chunk.fingerprint_sha256,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
        )
