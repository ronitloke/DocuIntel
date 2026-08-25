"""Add PostgreSQL full-text search representation and GIN index."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003_module5_search"
down_revision: Union[str, None] = "0002_module4_chunking_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a generated English search vector covering chunk text and headings."""

    op.add_column(
        "chunks",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english'::regconfig, "
                "coalesce(text, '') || ' ' || coalesce(section_heading, ''))",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunks_search_vector_gin",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Remove the Module 5 full-text representation and index."""

    op.drop_index("ix_chunks_search_vector_gin", table_name="chunks")
    op.drop_column("chunks", "search_vector")
