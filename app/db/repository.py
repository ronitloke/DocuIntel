"""Repository/data-access operations for persisted documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.core.exceptions import (
    ConversationNotFoundError,
    ConversationPersistenceError,
    DatabaseUnavailableError,
    DocumentPersistenceError,
    DuplicateDocumentError,
)
from app.db.models import (
    Chunk,
    Conversation,
    ConversationMessage,
    Document,
    DocumentTableRecord,
    LayoutElement,
    Page,
)
from app.services.chunking.structure_aware import ChunkDraft
from app.db.session import Database
from app.models.documents import (
    DocumentIngestionResponse,
    DocumentStatus,
    ExtractedTable,
    LayoutElement as LayoutElementSchema,
)
from app.models.conversations import MessageRole
from app.models.search import SearchFilters


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Explicit search projection that never selects the raw embedding vector."""

    chunk_id: UUID
    document_id: UUID
    original_filename: str
    sequence_number: int
    text: str
    section_heading: str | None
    start_page: int | None
    end_page: int | None
    content_type: str | None
    contains_ocr: bool
    score: float


@dataclass(frozen=True, slots=True)
class EvaluationDocumentRecord:
    """Small read-only document inventory projection used by evaluation preflight."""

    document_id: UUID
    original_filename: str
    is_indexed: bool
    chunk_count: int


@dataclass(frozen=True, slots=True)
class PersistedTableRecord:
    """Detached structured-table projection used by analysis and table queries."""

    table_id: UUID
    document_id: UUID
    original_filename: str
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]


class DocumentRepository:
    """Keep SQLAlchemy access out of routes and PDF/OCR services."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def persist_ingestion(self, response: DocumentIngestionResponse) -> None:
        """Persist one complete extraction in a single transaction."""

        session = self.database.session_factory()
        try:
            existing = session.scalar(
                select(Document.id).where(Document.checksum_sha256 == response.checksum_sha256)
            )
            if existing is not None:
                raise DuplicateDocumentError(
                    "An identical PDF has already been uploaded."
                )

            document = Document(
                id=response.document_id,
                original_filename=response.original_filename,
                stored_filename=response.stored_filename,
                file_size_bytes=response.file_size_bytes,
                mime_type=response.mime_type,
                checksum_sha256=response.checksum_sha256,
                page_count=response.page_count,
                status=response.status,
                title=response.metadata.title,
                author=response.metadata.author,
                subject=response.metadata.subject,
                keywords=response.metadata.keywords,
                creator=response.metadata.creator,
                producer=response.metadata.producer,
                creation_date=response.metadata.creation_date,
                modification_date=response.metadata.modification_date,
            )
            session.add(document)
            session.flush()

            for page_result in response.pages:
                page = Page(
                    document_id=document.id,
                    page_number=page_result.page_number,
                    extracted_text=page_result.text,
                    character_count=page_result.character_count,
                    has_native_text=page_result.has_native_text,
                    needs_ocr=page_result.needs_ocr,
                    extraction_method=page_result.extraction_method,
                    ocr_applied=page_result.ocr_applied,
                    ocr_success=page_result.ocr_success,
                    ocr_confidence=page_result.ocr_confidence,
                )
                session.add(page)
                session.flush()
                self._add_layout_elements(session, page.id, page_result.layout_elements)
                self._add_tables(session, page.id, page_result.tables)

            session.commit()
        except DuplicateDocumentError:
            session.rollback()
            raise
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError(
                "PostgreSQL is unavailable; the PDF was not persisted."
            ) from exc
        except IntegrityError as exc:
            session.rollback()
            raise DocumentPersistenceError(
                "The document could not be persisted because of a database constraint."
            ) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise DocumentPersistenceError(
                "The document could not be persisted safely."
            ) from exc
        finally:
            session.close()

    @staticmethod
    def _add_layout_elements(
        session: Session,
        page_id: UUID,
        elements: Iterable[LayoutElementSchema],
    ) -> None:
        """Add layout rows while preserving extraction order."""

        for sequence_order, element in enumerate(elements, start=1):
            bbox = element.bbox
            session.add(
                LayoutElement(
                    page_id=page_id,
                    sequence_order=sequence_order,
                    element_type=element.element_type,
                    text=element.text,
                    bbox_x0=bbox[0],
                    bbox_y0=bbox[1],
                    bbox_x1=bbox[2],
                    bbox_y1=bbox[3],
                    font_size=element.font_size,
                    is_bold=element.is_bold,
                )
            )

    @staticmethod
    def _add_tables(
        session: Session,
        page_id: UUID,
        tables: Iterable[ExtractedTable],
    ) -> None:
        """Add extracted table metadata and JSON rows."""

        for table in tables:
            bbox = table.bbox
            session.add(
                DocumentTableRecord(
                    page_id=page_id,
                    table_index=table.table_index,
                    bbox_x0=bbox[0],
                    bbox_y0=bbox[1],
                    bbox_x1=bbox[2],
                    bbox_y1=bbox[3],
                    headers=table.headers,
                    rows=table.rows,
                )
            )

    def list_documents(
        self,
        page: int,
        page_size: int,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        """Return one SQL-paginated document slice and its total count."""

        session = self.database.session_factory()
        try:
            conditions = [Document.status == status] if status is not None else []
            total = session.scalar(select(func.count(Document.id)).where(*conditions)) or 0
            documents = list(
                session.scalars(
                    select(Document)
                    .where(*conditions)
                    .order_by(Document.created_at.desc(), Document.id)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                ).all()
            )
            return documents, int(total)
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("Documents could not be listed safely.") from exc
        finally:
            session.close()

    def get_document(self, document_id: UUID) -> Document | None:
        """Return a document with pages and structure relationships loaded."""

        session = self.database.session_factory()
        try:
            statement = (
                select(Document)
                .where(Document.id == document_id)
                .options(
                    selectinload(Document.pages).selectinload(Page.layout_elements),
                    selectinload(Document.pages).selectinload(Page.tables),
                )
            )
            return session.scalars(statement).first()
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("The document could not be loaded safely.") from exc
        finally:
            session.close()

    def get_document_with_chunks(self, document_id: UUID) -> tuple[Document | None, list[Chunk]]:
        """Return one document and all its chunks in deterministic sequence order."""

        session = self.database.session_factory()
        try:
            statement = (
                select(Document)
                .where(Document.id == document_id)
                .options(selectinload(Document.chunks))
            )
            document = session.scalars(statement).first()
            if document is None:
                return None, []
            chunks = sorted(
                document.chunks,
                key=lambda chunk: (chunk.sequence_number, str(chunk.id)),
            )
            return document, chunks
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError(
                "Document analysis content could not be loaded safely."
            ) from exc
        finally:
            session.close()

    def get_document_with_chunks_and_tables(
        self,
        document_id: UUID,
    ) -> tuple[Document | None, list[Chunk], list[PersistedTableRecord]]:
        """Load one document's indexed chunks and existing structured tables."""

        session = self.database.session_factory()
        try:
            statement = (
                select(Document)
                .where(Document.id == document_id)
                .options(
                    selectinload(Document.chunks),
                    selectinload(Document.pages).selectinload(Page.tables),
                )
            )
            document = session.scalars(statement).first()
            if document is None:
                return None, [], []
            chunks = sorted(
                document.chunks,
                key=lambda chunk: (chunk.sequence_number, str(chunk.id)),
            )
            tables = self._table_records(document)
            return document, chunks, tables
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError(
                "Document analysis structure could not be loaded safely."
            ) from exc
        finally:
            session.close()

    @staticmethod
    def _table_records(document: Document) -> list[PersistedTableRecord]:
        """Project persisted JSONB table rows without retaining ORM session state."""

        records: list[PersistedTableRecord] = []
        for page in sorted(document.pages, key=lambda item: (item.page_number, str(item.id))):
            for table in sorted(page.tables, key=lambda item: (item.table_index, str(item.id))):
                headers = [str(value) for value in (table.headers or [])]
                rows = [
                    [str(value) for value in row]
                    for row in (table.rows or [])
                    if isinstance(row, list)
                ]
                records.append(
                    PersistedTableRecord(
                        table_id=table.id,
                        document_id=document.id,
                        original_filename=document.original_filename,
                        page_number=page.page_number,
                        table_index=table.table_index,
                        headers=headers,
                        rows=rows,
                    )
                )
        return records

    def list_document_tables(
        self,
        document_id: UUID,
    ) -> tuple[Document | None, list[PersistedTableRecord]]:
        """Return one document and all its detected tables in page order."""

        document, _chunks, tables = self.get_document_with_chunks_and_tables(document_id)
        return document, tables

    def get_document_table(
        self,
        document_id: UUID,
        table_id: UUID,
    ) -> tuple[Document | None, PersistedTableRecord | None]:
        """Return a table only when it belongs to the requested document."""

        document, tables = self.list_document_tables(document_id)
        for table in tables:
            if table.table_id == table_id:
                return document, table
        return document, None

    def document_exists(self, document_id: UUID) -> bool:
        """Check document existence without loading page text."""

        session = self.database.session_factory()
        try:
            return session.scalar(select(Document.id).where(Document.id == document_id)) is not None
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("The document could not be checked safely.") from exc
        finally:
            session.close()

    def indexed_document_ids(self, document_ids: Iterable[UUID]) -> set[UUID]:
        """Return requested document IDs that are ready and searchable."""

        unique_ids = list(dict.fromkeys(document_ids))
        if not unique_ids:
            return set()
        session = self.database.session_factory()
        try:
            rows = session.scalars(
                select(Document.id).where(
                    Document.id.in_(unique_ids),
                    Document.status == DocumentStatus.READY,
                    Document.is_indexed.is_(True),
                    Document.chunk_count > 0,
                )
            ).all()
            return set(rows)
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError(
                "The selected document scope could not be validated safely."
            ) from exc
        finally:
            session.close()

    def evaluation_document_inventory(self) -> list[EvaluationDocumentRecord]:
        """Return document labels and indexing counts without loading document content."""

        session = self.database.session_factory()
        try:
            rows = session.execute(
                select(
                    Document.id,
                    Document.original_filename,
                    Document.is_indexed,
                    Document.chunk_count,
                )
            ).all()
            return [
                EvaluationDocumentRecord(
                    document_id=row.id,
                    original_filename=row.original_filename,
                    is_indexed=row.is_indexed,
                    chunk_count=row.chunk_count,
                )
                for row in rows
            ]
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("Evaluation document inventory could not be loaded safely.") from exc
        finally:
            session.close()

    def list_pages(
        self,
        document_id: UUID,
        page: int,
        page_size: int,
    ) -> tuple[list[Page], int]:
        """Return one SQL-paginated page slice."""

        session = self.database.session_factory()
        try:
            total = session.scalar(
                select(func.count(Page.id)).where(Page.document_id == document_id)
            ) or 0
            pages = list(
                session.scalars(
                    select(Page)
                    .where(Page.document_id == document_id)
                    .order_by(Page.page_number)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                ).all()
            )
            return pages, int(total)
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("Pages could not be listed safely.") from exc
        finally:
            session.close()

    def get_page(self, document_id: UUID, page_number: int) -> Page | None:
        """Return one page with its layout and table relationships loaded."""

        session = self.database.session_factory()
        try:
            statement = (
                select(Page)
                .where(Page.document_id == document_id, Page.page_number == page_number)
                .options(selectinload(Page.layout_elements), selectinload(Page.tables))
            )
            return session.scalars(statement).first()
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("The page could not be loaded safely.") from exc
        finally:
            session.close()

    def replace_document_index(
        self,
        document_id: UUID,
        drafts: list[ChunkDraft],
        embeddings: list[list[float]],
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        """Replace all chunks and document indexing state in one transaction."""

        if len(drafts) != len(embeddings):
            raise DocumentPersistenceError("Chunk and embedding counts do not match.")
        session = self.database.session_factory()
        try:
            document = session.get(Document, document_id)
            if document is None:
                raise DocumentPersistenceError("The requested document was not found.")
            session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            for sequence_number, (draft, embedding) in enumerate(
                zip(drafts, embeddings, strict=True), start=1
            ):
                if len(embedding) != embedding_dimension:
                    raise DocumentPersistenceError("An embedding has an unexpected dimension.")
                session.add(
                    Chunk(
                        document_id=document_id,
                        page_id=draft.page_id,
                        sequence_number=sequence_number,
                        text=draft.text,
                        section_heading=draft.section_heading,
                        character_count=draft.character_count,
                        token_count=draft.token_count,
                        start_page=draft.start_page,
                        end_page=draft.end_page,
                        content_type=draft.content_type,
                        contains_ocr=draft.contains_ocr,
                        embedding=embedding,
                        embedding_model=embedding_model,
                        embedding_dimension=embedding_dimension,
                        fingerprint_sha256=draft.fingerprint_sha256,
                    )
                )
            document.is_indexed = True
            document.indexed_at = func.now()
            document.chunk_count = len(drafts)
            document.embedding_model = embedding_model
            document.embedding_dimension = embedding_dimension
            session.commit()
        except DocumentPersistenceError:
            session.rollback()
            raise
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError("PostgreSQL is unavailable; the index was not replaced.") from exc
        except IntegrityError as exc:
            session.rollback()
            raise DocumentPersistenceError(
                "The document index could not be persisted because of a database constraint."
            ) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise DocumentPersistenceError("The document index could not be persisted safely.") from exc
        finally:
            session.close()

    def list_chunks(
        self,
        document_id: UUID,
        page: int,
        page_size: int,
    ) -> tuple[list[Chunk], int]:
        """Return one paginated chunk slice without exposing vectors at the API layer."""

        session = self.database.session_factory()
        try:
            total = session.scalar(
                select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
            ) or 0
            chunks = list(
                session.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == document_id)
                    .order_by(Chunk.sequence_number)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                ).all()
            )
            return chunks, int(total)
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("Chunks could not be listed safely.") from exc
        finally:
            session.close()

    def get_chunk(self, document_id: UUID, chunk_id: UUID) -> Chunk | None:
        """Return one chunk only when it belongs to the requested document."""

        session = self.database.session_factory()
        try:
            return session.scalar(
                select(Chunk).where(Chunk.id == chunk_id, Chunk.document_id == document_id)
            )
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("The chunk could not be loaded safely.") from exc
        finally:
            session.close()

    def semantic_search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: SearchFilters | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchCandidate]:
        """Rank indexed chunks by pgvector cosine similarity inside PostgreSQL."""

        session = self.database.session_factory()
        try:
            distance = Chunk.embedding.cosine_distance(query_embedding)
            similarity = (1 - distance).label("score")
            statement = (
                select(*self._search_columns(), similarity)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    Document.is_indexed.is_(True),
                    Chunk.embedding.is_not(None),
                    *self._search_conditions(filters),
                )
                .order_by(distance, Chunk.document_id, Chunk.sequence_number, Chunk.id)
                .limit(limit)
            )
            if min_similarity is not None:
                statement = statement.where((1 - distance) >= min_similarity)
            rows = session.execute(statement).mappings().all()
            return [self._candidate_from_row(row) for row in rows]
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable during semantic search.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("Semantic search could not be completed safely.") from exc
        finally:
            session.close()

    def keyword_search(
        self,
        query: str,
        limit: int,
        text_search_config: str,
        filters: SearchFilters | None = None,
    ) -> list[SearchCandidate]:
        """Rank chunks with PostgreSQL full-text search and the generated GIN vector."""

        session = self.database.session_factory()
        try:
            tsquery = func.websearch_to_tsquery(text_search_config, query)
            keyword_score = func.ts_rank_cd(Chunk.search_vector, tsquery).label("score")
            statement = (
                select(*self._search_columns(), keyword_score)
                .join(Document, Document.id == Chunk.document_id)
                .where(
                    Document.is_indexed.is_(True),
                    Chunk.search_vector.op("@@")(tsquery),
                    *self._search_conditions(filters),
                )
                .order_by(keyword_score.desc(), Chunk.document_id, Chunk.sequence_number, Chunk.id)
                .limit(limit)
            )
            rows = session.execute(statement).mappings().all()
            return [self._candidate_from_row(row) for row in rows]
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable during keyword search.") from exc
        except SQLAlchemyError as exc:
            raise DocumentPersistenceError("Keyword search could not be completed safely.") from exc
        finally:
            session.close()

    @staticmethod
    def _search_columns() -> tuple[object, ...]:
        """Select only public result fields, leaving vectors in PostgreSQL."""

        return (
            Chunk.id.label("chunk_id"),
            Chunk.document_id.label("document_id"),
            Document.original_filename.label("original_filename"),
            Chunk.sequence_number.label("sequence_number"),
            Chunk.text.label("text"),
            Chunk.section_heading.label("section_heading"),
            Chunk.start_page.label("start_page"),
            Chunk.end_page.label("end_page"),
            Chunk.content_type.label("content_type"),
            Chunk.contains_ocr.label("contains_ocr"),
        )

    @staticmethod
    def _search_conditions(filters: SearchFilters | None) -> list[object]:
        """Translate optional filters into SQL predicates before ranking."""

        if filters is None:
            return []
        conditions: list[object] = []
        if filters.document_ids is not None:
            conditions.append(Chunk.document_id.in_(filters.document_ids))
        if filters.content_types is not None:
            conditions.append(Chunk.content_type.in_(filters.content_types))
        if filters.contains_ocr is not None:
            conditions.append(Chunk.contains_ocr == filters.contains_ocr)
        if filters.page_start is not None:
            conditions.append(or_(Chunk.end_page.is_(None), Chunk.end_page >= filters.page_start))
        if filters.page_end is not None:
            conditions.append(or_(Chunk.start_page.is_(None), Chunk.start_page <= filters.page_end))
        return conditions

    @staticmethod
    def _candidate_from_row(row: Mapping[str, object]) -> SearchCandidate:
        """Convert an explicit SQL mapping into a detached search candidate."""

        return SearchCandidate(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            original_filename=row["original_filename"],
            sequence_number=row["sequence_number"],
            text=row["text"],
            section_heading=row["section_heading"],
            start_page=row["start_page"],
            end_page=row["end_page"],
            content_type=row["content_type"],
            contains_ocr=row["contains_ocr"],
            score=float(row["score"]),
        )

    def delete_document(self, document_id: UUID) -> str | None:
        """Delete one document and database-owned dependents transactionally."""

        session = self.database.session_factory()
        try:
            document = session.get(Document, document_id)
            if document is None:
                return None
            stored_filename = document.stored_filename
            session.delete(document)
            session.commit()
            return stored_filename
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise DocumentPersistenceError("The document could not be deleted safely.") from exc
        finally:
            session.close()


class ConversationRepository:
    """Keep conversation and message persistence behind the existing repository layer."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_conversation(self, title: str | None = None) -> Conversation:
        """Create an empty conversation with optional user-supplied title."""

        session = self.database.session_factory()
        try:
            conversation = Conversation(title=title)
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise ConversationPersistenceError(
                "The conversation could not be created safely."
            ) from exc
        finally:
            session.close()

    def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Return conversation metadata without loading unbounded history."""

        session = self.database.session_factory()
        try:
            return session.get(Conversation, conversation_id)
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise ConversationPersistenceError(
                "The conversation could not be loaded safely."
            ) from exc
        finally:
            session.close()

    def require_conversation(self, conversation_id: UUID) -> Conversation:
        """Load a conversation or raise the controlled public 404 error."""

        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("The requested conversation was not found.")
        return conversation

    def list_conversations(self, limit: int = 100) -> list[Conversation]:
        """List recent sessions deterministically by most recently updated first."""

        session = self.database.session_factory()
        try:
            return list(
                session.scalars(
                    select(Conversation)
                    .order_by(
                        Conversation.updated_at.desc(),
                        Conversation.created_at.desc(),
                        Conversation.id,
                    )
                    .limit(limit)
                ).all()
            )
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise ConversationPersistenceError(
                "Conversations could not be listed safely."
            ) from exc
        finally:
            session.close()

    def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete a session; PostgreSQL cascades its messages."""

        session = self.database.session_factory()
        try:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return False
            session.delete(conversation)
            session.commit()
            return True
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise ConversationPersistenceError(
                "The conversation could not be deleted safely."
            ) from exc
        finally:
            session.close()

    def append_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
    ) -> ConversationMessage:
        """Append one message with a transactionally allocated sequence number."""

        normalized_content = content.strip()
        if not normalized_content:
            raise ConversationPersistenceError("Conversation message content cannot be blank.")

        session = self.database.session_factory()
        try:
            conversation = session.scalar(
                select(Conversation)
                .where(Conversation.id == conversation_id)
                .with_for_update()
            )
            if conversation is None:
                raise ConversationNotFoundError("The requested conversation was not found.")
            current_max = session.scalar(
                select(func.max(ConversationMessage.sequence_number)).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            ) or 0
            message = ConversationMessage(
                id=uuid4(),
                conversation_id=conversation_id,
                role=role,
                content=normalized_content,
                sequence_number=int(current_max) + 1,
            )
            session.add(message)
            conversation.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(message)
            return message
        except ConversationNotFoundError:
            session.rollback()
            raise
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except IntegrityError as exc:
            session.rollback()
            raise ConversationPersistenceError(
                "The conversation message could not be persisted safely."
            ) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise ConversationPersistenceError(
                "The conversation message could not be persisted safely."
            ) from exc
        finally:
            session.close()

    def list_messages(self, conversation_id: UUID) -> list[ConversationMessage]:
        """Return all messages in stable chronological sequence order."""

        session = self.database.session_factory()
        try:
            return list(
                session.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id == conversation_id)
                    .order_by(
                        ConversationMessage.sequence_number,
                        ConversationMessage.id,
                    )
                ).all()
            )
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise ConversationPersistenceError(
                "Conversation messages could not be listed safely."
            ) from exc
        finally:
            session.close()

    def list_recent_messages(
        self,
        conversation_id: UUID,
        *,
        max_messages: int,
        max_chars: int,
    ) -> list[ConversationMessage]:
        """Return the newest bounded history in chronological order."""

        if max_messages <= 0 or max_chars <= 0:
            return []
        session = self.database.session_factory()
        try:
            newest = list(
                session.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id == conversation_id)
                    .order_by(
                        ConversationMessage.sequence_number.desc(),
                        ConversationMessage.id.desc(),
                    )
                    .limit(max_messages)
                ).all()
            )
            selected: list[ConversationMessage] = []
            used_chars = 0
            for message in newest:
                if used_chars + len(message.content) > max_chars:
                    break
                selected.append(message)
                used_chars += len(message.content)
            selected.sort(key=lambda message: (message.sequence_number, message.id))
            return selected
        except OperationalError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            raise ConversationPersistenceError(
                "Conversation history could not be loaded safely."
            ) from exc
        finally:
            session.close()

    def set_title_if_empty(self, conversation_id: UUID, title: str) -> None:
        """Set a deterministic first-question title without overwriting user metadata."""

        normalized_title = title.strip()[:512]
        if not normalized_title:
            return
        session = self.database.session_factory()
        try:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError("The requested conversation was not found.")
            if conversation.title is None:
                conversation.title = normalized_title
                conversation.updated_at = datetime.now(UTC)
                session.commit()
        except ConversationNotFoundError:
            session.rollback()
            raise
        except OperationalError as exc:
            session.rollback()
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise ConversationPersistenceError(
                "The conversation title could not be updated safely."
            ) from exc
        finally:
            session.close()
