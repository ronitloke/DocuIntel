"""Thin API route for evidence-first document comparison."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings
from app.db.repository import DocumentRepository
from app.db.session import Database
from app.models.comparison import ComparisonRequest, ComparisonResponse
from app.services.comparison.service import ComparisonService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_comparison_service(request: Request) -> ComparisonService:
    """Build the comparison orchestrator from existing application resources."""

    database: Database | None = request.app.state.database
    repository = DocumentRepository(database) if database is not None else None
    settings: Settings = request.app.state.settings
    return ComparisonService(
        repository=repository,
        ollama_client=request.app.state.ollama_client,
        settings=settings,
    )


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    tags=["comparison"],
    summary="Compare two ready indexed documents",
)
async def compare_documents(
    request_body: ComparisonRequest,
    service: ComparisonService = Depends(get_comparison_service),
) -> ComparisonResponse:
    """Compare an explicitly selected base/target pair without persisting a result."""

    return await service.compare(request_body)
