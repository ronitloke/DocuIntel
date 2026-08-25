"""Configuration for the HTTP-only Streamlit presentation layer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendSettings(BaseSettings):
    """Environment-backed settings used by the Streamlit app."""

    api_base_url: str = Field(
        default="http://127.0.0.1:8001",
        validation_alias="DOCUINTEL_API_BASE_URL",
    )
    api_timeout_seconds: float = Field(
        default=180.0,
        validation_alias="DOCUINTEL_API_TIMEOUT_SECONDS",
    )
    max_upload_size_mb: int = Field(default=25, gt=0, validation_alias="MAX_UPLOAD_SIZE_MB")
    ollama_model: str = Field(default="llama3.2:3b", validation_alias="OLLAMA_MODEL")
    e5_results_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "evaluation" / "results" / "e5" / "final_baseline_20260821_final",
        validation_alias="DOCUINTEL_E5_RESULTS_DIR",
    )

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("DOCUINTEL_API_BASE_URL must start with http:// or https://.")
        return value

    @field_validator("api_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("DOCUINTEL_API_TIMEOUT_SECONDS must be greater than zero.")
        return value

    @field_validator("ollama_model")
    @classmethod
    def validate_ollama_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OLLAMA_MODEL must not be empty.")
        return value

    @field_validator("e5_results_directory")
    @classmethod
    def validate_e5_results_directory(cls, value: Path) -> Path:
        if not str(value).strip():
            raise ValueError("DOCUINTEL_E5_RESULTS_DIR must not be empty.")
        return value.expanduser()


@lru_cache(maxsize=1)
def get_settings() -> FrontendSettings:
    """Return cached frontend settings for the current process."""

    return FrontendSettings()
