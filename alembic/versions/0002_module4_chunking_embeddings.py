"""Add Module 4 chunk metadata and document indexing state."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_module4_chunking_embeddings"
down_revision: Union[str, None] = "0001_module3_document_storage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add indexing state and structure/provenance metadata without rewriting Module 3."""

    op.add_column(
        "documents",
        sa.Column("is_indexed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("documents", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "documents",
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("documents", sa.Column("embedding_model", sa.String(length=512), nullable=True))
    op.add_column("documents", sa.Column("embedding_dimension", sa.Integer(), nullable=True))

    op.add_column("chunks", sa.Column("start_page", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("end_page", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("content_type", sa.String(length=16), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("contains_ocr", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("chunks", sa.Column("embedding_model", sa.String(length=512), nullable=True))
    op.add_column("chunks", sa.Column("embedding_dimension", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("fingerprint_sha256", sa.String(length=64), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_chunks_document_page_range", "chunks", ["document_id", "start_page", "end_page"]
    )
    op.create_index(
        "ix_chunks_document_fingerprint", "chunks", ["document_id", "fingerprint_sha256"]
    )


def downgrade() -> None:
    """Remove Module 4 columns while retaining the Module 3 schema."""

    op.drop_index("ix_chunks_document_fingerprint", table_name="chunks")
    op.drop_index("ix_chunks_document_page_range", table_name="chunks")
    for column in (
        "updated_at",
        "fingerprint_sha256",
        "embedding_dimension",
        "embedding_model",
        "contains_ocr",
        "content_type",
        "end_page",
        "start_page",
    ):
        op.drop_column("chunks", column)
    for column in (
        "embedding_dimension",
        "embedding_model",
        "chunk_count",
        "indexed_at",
        "is_indexed",
    ):
        op.drop_column("documents", column)
