"""DocuIntel FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.structured import router as structured_router
from app.api.routes.comparison import router as comparison_router
from app.api.routes.privacy import router as privacy_router
from app.api.routes.rag import router as rag_router
from app.api.routes.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentIngestionError
from app.core.logging import configure_logging
from app.core.version import VERSION
from app.db.session import Database, create_database
from app.services.embeddings.sentence_transformer import EmbeddingService
from app.services.ocr.tesseract_ocr import OCRService
from app.services.llm.ollama import OllamaClient
from app.services.reranking.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    storage_directory: Path | None = None,
    ocr_service: OCRService | None = None,
    database: Database | None = None,
    embedding_service: EmbeddingService | None = None,
    reranker: CrossEncoderReranker | None = None,
    ollama_client: OllamaClient | None = None,
) -> FastAPI:
    """Create and configure the DocuIntel FastAPI application."""

    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)
    app_database = database
    if app_database is None and storage_directory is None:
        try:
            app_database = create_database(app_settings)
        except Exception as exc:
            logger.error(
                "PostgreSQL resource initialization failed dependency=postgresql "
                "error_type=%s continue=true readiness=false",
                type(exc).__name__,
            )
            app_database = None
    if app_database is None and storage_directory is None:
        logger.warning(
            "PostgreSQL is not configured dependency=postgresql "
            "continue=true readiness=false"
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting %s version=%s environment=%s",
            app_settings.app_name,
            VERSION,
            app_settings.app_env,
        )
        yield
        if app_database is not None:
            app_database.engine.dispose()
        logger.info("Shutting down %s", app_settings.app_name)

    application = FastAPI(
        title=app_settings.app_name,
        version=VERSION,
        description="Bootstrap API for the DocuIntel intelligent document platform.",
        lifespan=lifespan,
    )

    @application.exception_handler(DocumentIngestionError)
    async def handle_document_ingestion_error(
        _request: Request,
        exc: DocumentIngestionError,
    ) -> JSONResponse:
        """Map dependency-time controlled errors to their safe public response."""

        return JSONResponse(status_code=exc.status_code, content={"detail": exc.public_message})

    application.state.settings = app_settings
    application.state.pdf_storage_directory = storage_directory
    application.state.ocr_service = ocr_service
    application.state.database = app_database
    # The service is cheap to construct; its Sentence Transformer model remains
    # lazy and is loaded only when an indexing request actually needs it.
    application.state.embedding_service = embedding_service or EmbeddingService(
        settings=app_settings
    )
    # The CrossEncoder is lazy; constructing the service does not download or
    # load the model during application startup.
    application.state.reranker = reranker or CrossEncoderReranker(settings=app_settings)
    # Ollama is an external HTTP service; constructing this client does not
    # start a process or make a network call during FastAPI startup.
    application.state.ollama_client = ollama_client or OllamaClient(settings=app_settings)
    # Existing isolated unit callers can exercise extraction without PostgreSQL;
    # the default application instance requires persistence for uploads.
    application.state.persistence_required = storage_directory is None
    application.include_router(health_router)
    application.include_router(documents_router, prefix="/api/v1/documents")
    application.include_router(analysis_router, prefix="/api/v1")
    application.include_router(structured_router, prefix="/api/v1")
    application.include_router(comparison_router, prefix="/api/v1")
    application.include_router(privacy_router, prefix="/api/v1")
    application.include_router(search_router, prefix="/api/v1")
    application.include_router(rag_router, prefix="/api/v1")
    application.include_router(conversations_router, prefix="/api/v1")
    return application


app = create_app()
