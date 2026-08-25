"""Liveness and PostgreSQL readiness endpoints."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.version import VERSION
from app.db.health import check_database

router = APIRouter()


class HealthResponse(BaseModel):
    """Response returned by the service health endpoint."""

    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Response returned by the database readiness endpoint."""

    status: str
    service: str
    version: str
    database: str


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Report whether the API process is running."""

    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=VERSION,
    )


@router.get("/ready", response_model=ReadyResponse, tags=["system"])
def readiness_check(request: Request) -> ReadyResponse | JSONResponse:
    """Report whether the configured PostgreSQL database accepts a query."""

    settings = request.app.state.settings
    healthy = check_database(request.app.state.database)
    payload = ReadyResponse(
        status="healthy" if healthy else "unready",
        service=settings.app_name,
        version=VERSION,
        database="healthy" if healthy else "unavailable",
    )
    if healthy:
        return payload
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(mode="json"),
    )
