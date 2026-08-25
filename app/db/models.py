"""Persistent PostgreSQL models for documents and extracted structure."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SqlEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Computed

from app.db.base import Base
from app.models.conversations import MessageRole
from app.models.documents import DocumentStatus


class Document(Base):
    """Authoritative metadata record for one uploaded PDF."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    original_filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        SqlEnum(
            DocumentStatus,
            name="document_status",
            values_callable=lambda values: [item.value for item in values],
        ),
        nullable=False,
        default=DocumentStatus.UPLOADED,
    )
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    author: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    keywords: Mapped[str | None] = mapped_column(Text, nullable=True)
    creator: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    producer: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    creation_date: Mapped[str | None] = mapped_column(String(255), nullable=True)
    modification_date: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding_model: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    pages: Mapped[list[Page]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Page.page_number",
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("file_size_bytes > 0", name="ck_documents_file_size_positive"),
        CheckConstraint("page_count >= 0", name="ck_documents_page_count_nonnegative"),
        Index("ix_documents_status_created_at", "status", "created_at"),
    )


class Page(Base):
    """One human-numbered PDF page and its extraction/OCR state."""

    __tablename__ = "pages"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    has_native_text: Mapped[bool] = mapped_column(Boolean, nullable=False)
    needs_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(32), nullable=False)
    ocr_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ocr_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="pages")
    layout_elements: Mapped[list[LayoutElement]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="LayoutElement.sequence_order",
    )
    tables: Mapped[list[DocumentTableRecord]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentTableRecord.table_index",
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="page", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_pages_document_page_number"),
        CheckConstraint("page_number > 0", name="ck_pages_page_number_positive"),
        CheckConstraint("character_count >= 0", name="ck_pages_character_count_nonnegative"),
        CheckConstraint(
            "ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 100)",
            name="ck_pages_ocr_confidence_range",
        ),
    )


class LayoutElement(Base):
    """Heuristic layout element associated with one page."""

    __tablename__ = "layout_elements"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence_order: Mapped[int] = mapped_column(Integer, nullable=False)
    element_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_x0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    font_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_bold: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    page: Mapped[Page] = relationship(back_populates="layout_elements")


class DocumentTableRecord(Base):
    """Native table metadata with maintainable JSON headers and rows."""

    __tablename__ = "document_tables"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    page_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    headers: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    rows: Mapped[list[list[str]]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    page: Mapped[Page] = relationship(back_populates="tables")

    __table_args__ = (
        UniqueConstraint("page_id", "table_index", name="uq_document_tables_page_table_index"),
        CheckConstraint("table_index > 0", name="ck_document_tables_index_positive"),
    )


class Chunk(Base):
    """A structure-aware document chunk and its optional pgvector embedding."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("pages.id", ondelete="SET NULL"), nullable=True, index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section_heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    contains_ocr: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fingerprint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR(),
        Computed(
            "to_tsvector('english'::regconfig, "
            "coalesce(text, '') || ' ' || coalesce(section_heading, ''))",
            persisted=True,
        ),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    page: Mapped[Page | None] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "sequence_number", name="uq_chunks_document_sequence"),
        CheckConstraint("sequence_number > 0", name="ck_chunks_sequence_positive"),
        CheckConstraint("character_count >= 0", name="ck_chunks_character_count_nonnegative"),
        CheckConstraint(
            "start_page IS NULL OR start_page > 0", name="ck_chunks_start_page_positive"
        ),
        CheckConstraint(
            "end_page IS NULL OR end_page > 0", name="ck_chunks_end_page_positive"
        ),
        CheckConstraint(
            "end_page IS NULL OR start_page IS NULL OR end_page >= start_page",
            name="ck_chunks_page_range_valid",
        ),
        Index("ix_chunks_document_page_range", "document_id", "start_page", "end_page"),
        Index("ix_chunks_document_fingerprint", "document_id", "fingerprint_sha256"),
    )


class DocumentVersion(Base):
    """Simple version chain for future document comparison."""

    __tablename__ = "document_versions"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_version_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True
    )
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="versions")
    previous_version: Mapped[DocumentVersion | None] = relationship(
        remote_side=[id], back_populates="next_versions"
    )
    next_versions: Mapped[list[DocumentVersion]] = relationship(back_populates="previous_version")

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_versions_document_number"),
        CheckConstraint("version_number > 0", name="ck_document_versions_number_positive"),
    )


class Conversation(Base):
    """A persisted multi-turn question-answer session."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ConversationMessage.sequence_number",
    )

    __table_args__ = (Index("ix_conversations_updated_at", "updated_at"),)


class ConversationMessage(Base):
    """One ordered user or assistant message belonging to a conversation."""

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[MessageRole] = mapped_column(
        SqlEnum(
            MessageRole,
            name="conversation_message_role",
            values_callable=lambda values: [item.value for item in values],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uq_messages_conversation_sequence",
        ),
        CheckConstraint("sequence_number > 0", name="ck_messages_sequence_positive"),
        CheckConstraint("length(trim(content)) > 0", name="ck_messages_content_nonempty"),
        Index(
            "ix_messages_conversation_sequence",
            "conversation_id",
            "sequence_number",
        ),
    )
