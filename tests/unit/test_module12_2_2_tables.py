"""Regression tests for reliable natural-language table planning."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.exceptions import TableQueryValidationError
from app.db.repository import PersistedTableRecord
from app.core.config import Settings
from app.models.documents import DocumentStatus
from app.models.structured import TableQueryOperation, TableQueryRequest

from app.services.tables.query import TableQueryService


DOCUMENT_ID = uuid4()
TABLE_ID = uuid4()


def make_table(*, headers: list[str], rows: list[list[str]]) -> PersistedTableRecord:
    return PersistedTableRecord(
        table_id=TABLE_ID,
        document_id=DOCUMENT_ID,
        original_filename="layout_table_module4.pdf",
        page_number=1,
        table_index=1,
        headers=headers,
        rows=rows,
    )


class TableRepository:
    def __init__(self, table: PersistedTableRecord) -> None:
        self.document = SimpleNamespace(
            id=DOCUMENT_ID,
            original_filename="layout_table_module4.pdf",
            status=DocumentStatus.READY,
            is_indexed=True,
        )
        self.table = table

    def list_document_tables(self, document_id):
        assert document_id == DOCUMENT_ID
        return self.document, [self.table]


class FixedPlanProvider:
    model = "test-model"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        return self.response


def service(provider: FixedPlanProvider, table: PersistedTableRecord) -> TableQueryService:
    return TableQueryService(TableRepository(table), provider, Settings(table_query_max_rows=100))


def real_table() -> PersistedTableRecord:
    """Create the same one-row table used by local acceptance testing."""

    return make_table(
        headers=["Product", "Qty", "Price"],
        rows=[["Laptop", "3", "2400"]],
    )


def run_natural_language(question: str, provider: FixedPlanProvider | None = None):
    """Run one natural-language query against the acceptance-shaped table."""

    selected_provider = provider or FixedPlanProvider({})
    response = asyncio.run(
        service(selected_provider, real_table()).query(
            DOCUMENT_ID,
            TABLE_ID,
            TableQueryRequest(question=question),
        )
    )
    return response, selected_provider


@pytest.mark.parametrize(
    ("question", "expected_value", "expected_noun"),
    [
        ("How many products are in the table?", 1, "product"),
        ("How many rows are in the table?", 1, "row"),
        ("How many records are there?", 1, "record"),
    ],
)
def test_obvious_count_questions_bypass_invalid_ollama_and_count_data_rows(
    question: str,
    expected_value: int,
    expected_noun: str,
) -> None:
    response, provider = run_natural_language(question)

    assert provider.calls == 0
    assert response.plan.operation is TableQueryOperation.COUNT
    assert response.plan.target_column is None
    assert response.result.value == expected_value
    assert f"{expected_value} {expected_noun}" in response.answer
    assert response.sources[0].source_id == "T1"
    assert response.sources[0].row_indices == [1]


@pytest.mark.parametrize(
    ("question", "target", "expected"),
    [
        ("What is the total quantity?", "Qty", "3"),
        ("What is the total price?", "Price", "2400"),
        ("What is the average quantity?", "Qty", "3"),
    ],
)
def test_obvious_numeric_aggregates_resolve_safe_measure_columns(
    question: str,
    target: str,
    expected: str,
) -> None:
    response, provider = run_natural_language(question)

    assert provider.calls == 0
    assert response.plan.target_column == target
    assert response.result.value == expected
    assert response.sources[0].row_indices == [1]


@pytest.mark.parametrize(
    ("question", "operation"),
    [
        ("Which product has the highest quantity?", TableQueryOperation.MAX),
        ("Which product has the lowest quantity?", TableQueryOperation.MIN),
    ],
)
def test_rank_shortcuts_preserve_max_min_and_quantity_alias(
    question: str,
    operation: TableQueryOperation,
) -> None:
    response, provider = run_natural_language(question)

    assert provider.calls == 0
    assert response.plan.operation is operation
    assert response.plan.target_column == "Qty"
    assert response.result.rows == [{"Product": "Laptop", "Qty": "3"}]
    assert response.answer.endswith("[T1]")


def test_unknown_salary_measure_never_defaults_to_product() -> None:
    with pytest.raises(TableQueryValidationError, match='requested column "employee salary"'):
        run_natural_language("Which product has the highest employee salary?")


def test_genuinely_ambiguous_malformed_provider_output_still_fails_closed() -> None:
    malformed = FixedPlanProvider({"operation": "not-a-supported-operation"})

    with pytest.raises(TableQueryValidationError, match="invalid table query plan"):
        run_natural_language("Show the best item.", malformed)

    assert malformed.calls == 1
