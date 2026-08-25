"""API contract tests for Module 12.2 endpoints."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.routes.structured import get_structured_extraction_service, get_table_query_service
from app.core.exceptions import OllamaServiceError
from app.main import create_app
from app.models.structured import (
    ExtractionStatus,
    StructuredExtractionFieldResult,
    StructuredExtractionResponse,
    TableInventoryItem,
    TableInventoryResponse,
)


DOCUMENT_ID = uuid4()
TABLE_ID = uuid4()
CHUNK_ID = uuid4()


class FixedExtractionService:
    """Dependency double for the extraction endpoint."""

    async def extract(self, document_id, request):
        return StructuredExtractionResponse(
            document_id=document_id,
            filename="manual.pdf",
            model="test-model",
            fields=[
                StructuredExtractionFieldResult(
                    field=request.fields[0].name,
                    value=None,
                    status=ExtractionStatus.NOT_FOUND,
                    sources=[],
                )
            ],
            sources=[],
            evidence_loading_time_ms=1,
            generation_time_ms=1,
            validation_time_ms=1,
            total_time_ms=3,
        )


class FixedTableService:
    """Dependency double for table inventory."""

    def inventory(self, document_id):
        return TableInventoryResponse(
            document_id=document_id,
            filename="table.pdf",
            tables=[
                TableInventoryItem(
                    table_id=TABLE_ID,
                    document_id=document_id,
                    filename="table.pdf",
                    page_number=1,
                    table_index=1,
                    row_count=1,
                    column_count=2,
                    headers=["Product", "Revenue"],
                )
            ],
        )


def test_valid_extraction_and_table_inventory_api_contract(tmp_path) -> None:
    application = create_app(storage_directory=tmp_path / "uploads")
    application.dependency_overrides[get_structured_extraction_service] = lambda: FixedExtractionService()
    application.dependency_overrides[get_table_query_service] = lambda: FixedTableService()

    with TestClient(application) as client:
        extraction = client.post(
            f"/api/v1/documents/{DOCUMENT_ID}/extract",
            json={"fields": [{"name": "employee_name", "type": "string"}]},
        )
        tables = client.get(f"/api/v1/documents/{DOCUMENT_ID}/tables")

    assert extraction.status_code == 200
    assert extraction.json()["fields"][0]["status"] == "not_found"
    assert tables.status_code == 200
    assert tables.json()["tables"][0]["table_id"] == str(TABLE_ID)


@pytest.mark.parametrize(
    "payload",
    [
        {"fields": []},
        {"fields": [{"name": "a", "type": "python_eval"}]},
        {"fields": [{"name": "a", "type": "string"}, {"name": "a", "type": "string"}]},
        {"fields": [{"name": "a", "type": "string", "code": "DROP TABLE"}]},
    ],
)
def test_invalid_extraction_request_is_rejected_before_service_call(tmp_path, payload) -> None:
    application = create_app(storage_directory=tmp_path / "uploads")
    application.dependency_overrides[get_structured_extraction_service] = lambda: FixedExtractionService()

    with TestClient(application) as client:
        response = client.post(f"/api/v1/documents/{DOCUMENT_ID}/extract", json=payload)

    assert response.status_code == 422


def test_provider_unavailable_is_mapped_to_controlled_error(tmp_path) -> None:
    class UnavailableExtractionService:
        async def extract(self, document_id, request):
            raise OllamaServiceError("The local Ollama service is unavailable.")

    application = create_app(storage_directory=tmp_path / "uploads")
    application.dependency_overrides[get_structured_extraction_service] = lambda: UnavailableExtractionService()

    with TestClient(application) as client:
        response = client.post(
            f"/api/v1/documents/{DOCUMENT_ID}/extract",
            json={"fields": [{"name": "employee_name", "type": "string"}]},
        )

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]
