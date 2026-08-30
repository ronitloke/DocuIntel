"""Application settings loaded from environment variables and optional .env files."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the DocuIntel application."""

    app_name: str = "DocuIntel"
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str | None = None
    postgres_db: str = "docuintel"
    postgres_user: str = "docuintel"
    postgres_password: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, gt=0, le=65535)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = Field(default=120, gt=0)
    ollama_temperature: float = Field(default=0.1, ge=0, le=2)
    max_upload_size_mb: int = Field(default=25, gt=0)
    ocr_candidate_char_threshold: int = Field(default=20, ge=0)
    tesseract_cmd: str | None = None
    ocr_language: str = "eng"
    ocr_render_dpi: int = Field(default=300, gt=0)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    chunk_target_chars: int = Field(default=1200, gt=0)
    chunk_max_chars: int = Field(default=1800, gt=0)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    search_default_top_k: int = Field(default=5, gt=0)
    search_max_top_k: int = Field(default=50, gt=0)
    hybrid_rrf_k: int = Field(default=60, gt=0)
    search_candidate_multiplier: int = Field(default=4, gt=0)
    semantic_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    search_max_query_chars: int = Field(default=1000, gt=0)
    postgres_text_search_config: str = "english"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_max_length: int = Field(default=512, gt=0)
    rerank_candidate_count: int = Field(default=20, gt=0)
    rerank_candidate_multiplier: int = Field(default=4, gt=0)
    rerank_max_candidates: int = Field(default=200, gt=0)
    rerank_default_enabled: bool = False
    rag_default_top_k: int = Field(default=5, gt=0)
    rag_max_context_chars: int = Field(default=12000, gt=0)
    rag_max_selected_documents: int = Field(default=20, gt=0)
    rag_selected_document_candidate_multiplier: int = Field(default=4, gt=0)
    rag_max_selected_candidates: int = Field(default=100, gt=0)
    rag_history_max_messages: int = Field(default=10, gt=0)
    rag_history_max_chars: int = Field(default=6000, gt=0)
    summary_batch_max_chars: int = Field(default=6000, gt=0)
    summary_final_max_chars: int = Field(default=12000, gt=0)
    summary_grounding_enabled: bool = True
    summary_grounding_max_passes: int = Field(default=2, ge=1, le=2)
    structured_extraction_max_fields: int = Field(default=10, gt=0, le=50)
    structured_extraction_max_context_chars: int = Field(default=12000, gt=0)
    table_query_max_rows: int = Field(default=10000, gt=0)
    comparison_max_blocks: int = Field(default=1000, gt=0, le=10000)
    comparison_max_content_chars: int = Field(default=100000, gt=0)
    comparison_summary_max_chars: int = Field(default=12000, gt=0)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        """Require an HTTP(S) Ollama endpoint without embedding host assumptions."""

        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://.")
        return url

    @field_validator("ollama_model")
    @classmethod
    def validate_ollama_model(cls, value: str) -> str:
        """Reject an empty local model name."""

        model = value.strip()
        if not model:
            raise ValueError("ollama_model must contain a model name.")
        return model

    @property
    def resolved_database_url(self) -> str | None:
        """Return the configured PostgreSQL URL without inventing credentials."""

        if self.database_url and self.database_url.strip():
            return self.database_url.strip()
        if not self.postgres_password:
            return None

        username = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password, safe="")
        database = quote(self.postgres_db, safe="")
        host = self.postgres_host.strip()
        return (
            f"postgresql+psycopg://{username}:{password}@{host}:"
            f"{self.postgres_port}/{database}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
