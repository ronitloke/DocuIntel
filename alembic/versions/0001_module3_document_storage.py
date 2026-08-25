"""Create the Module 3 PostgreSQL, pgvector, and document schema."""

from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_module3_document_storage"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the authoritative document metadata schema."""

    bind = op.get_bind()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    status_enum = postgresql.ENUM(
        "uploaded", "processing", "ready", "failed", name="document_status", create_type=False
    )
    status_enum.create(bind, checkfirst=True)

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_filename", sa.String(length=1024), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("status", status_enum, nullable=False, server_default=sa.text("'uploaded'")),
        sa.Column("title", sa.String(length=1024), nullable=True),
        sa.Column("author", sa.String(length=1024), nullable=True),
        sa.Column("subject", sa.String(length=2048), nullable=True),
        sa.Column("keywords", sa.Text(), nullable=True),
        sa.Column("creator", sa.String(length=1024), nullable=True),
        sa.Column("producer", sa.String(length=1024), nullable=True),
        sa.Column("creation_date", sa.String(length=255), nullable=True),
        sa.Column("modification_date", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("file_size_bytes > 0", name="ck_documents_file_size_positive"),
        sa.CheckConstraint("page_count >= 0", name="ck_documents_page_count_nonnegative"),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("checksum_sha256", name="uq_documents_checksum_sha256"),
        sa.UniqueConstraint("stored_filename", name="uq_documents_stored_filename"),
    )
    op.create_index("ix_documents_status_created_at", "documents", ["status", "created_at"])

    op.create_table(
        "pages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("has_native_text", sa.Boolean(), nullable=False),
        sa.Column("needs_ocr", sa.Boolean(), nullable=False),
        sa.Column("extraction_method", sa.String(length=32), nullable=False),
        sa.Column("ocr_applied", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ocr_success", sa.Boolean(), nullable=True),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("page_number > 0", name="ck_pages_page_number_positive"),
        sa.CheckConstraint("character_count >= 0", name="ck_pages_character_count_nonnegative"),
        sa.CheckConstraint(
            "ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 100)",
            name="ck_pages_ocr_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE", name="fk_pages_document_id_documents"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_pages"),
        sa.UniqueConstraint("document_id", "page_number", name="uq_pages_document_page_number"),
    )
    op.create_index("ix_pages_document_id", "pages", ["document_id"])

    op.create_table(
        "layout_elements",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence_order", sa.Integer(), nullable=False),
        sa.Column("element_type", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("bbox_x0", sa.Float(), nullable=False),
        sa.Column("bbox_y0", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("font_size", sa.Float(), nullable=True),
        sa.Column("is_bold", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["page_id"], ["pages.id"], ondelete="CASCADE", name="fk_layout_elements_page_id_pages"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_layout_elements"),
    )
    op.create_index("ix_layout_elements_page_id", "layout_elements", ["page_id"])

    op.create_table(
        "document_tables",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("table_index", sa.Integer(), nullable=False),
        sa.Column("bbox_x0", sa.Float(), nullable=False),
        sa.Column("bbox_y0", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("headers", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rows", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("table_index > 0", name="ck_document_tables_index_positive"),
        sa.ForeignKeyConstraint(
            ["page_id"], ["pages.id"], ondelete="CASCADE", name="fk_document_tables_page_id_pages"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_tables"),
        sa.UniqueConstraint("page_id", "table_index", name="uq_document_tables_page_table_index"),
    )
    op.create_index("ix_document_tables_page_id", "document_tables", ["page_id"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section_heading", sa.Text(), nullable=True),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("sequence_number > 0", name="ck_chunks_sequence_positive"),
        sa.CheckConstraint("character_count >= 0", name="ck_chunks_character_count_nonnegative"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE", name="fk_chunks_document_id_documents"
        ),
        sa.ForeignKeyConstraint(
            ["page_id"], ["pages.id"], ondelete="SET NULL", name="fk_chunks_page_id_pages"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint("document_id", "sequence_number", name="uq_chunks_document_sequence"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_page_id", "chunks", ["page_id"])

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("previous_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("version_number > 0", name="ck_document_versions_number_positive"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE", name="fk_document_versions_document_id_documents"
        ),
        sa.ForeignKeyConstraint(
            ["previous_version_id"], ["document_versions.id"], ondelete="SET NULL", name="fk_document_versions_previous_version_id_document_versions"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_versions"),
        sa.UniqueConstraint("document_id", "version_number", name="uq_document_versions_document_number"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])


def downgrade() -> None:
    """Remove Module 3 tables while leaving pgvector available for other schemas."""

    op.drop_index("ix_document_versions_document_id", table_name="document_versions")
    op.drop_table("document_versions")
    op.drop_index("ix_chunks_page_id", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_document_tables_page_id", table_name="document_tables")
    op.drop_table("document_tables")
    op.drop_index("ix_layout_elements_page_id", table_name="layout_elements")
    op.drop_table("layout_elements")
    op.drop_index("ix_pages_document_id", table_name="pages")
    op.drop_table("pages")
    op.drop_index("ix_documents_status_created_at", table_name="documents")
    op.drop_table("documents")
    postgresql.ENUM(
        "uploaded", "processing", "ready", "failed", name="document_status"
    ).drop(op.get_bind(), checkfirst=True)
