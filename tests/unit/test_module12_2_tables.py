"""Deterministic tests for Module 12.2 structured table querying."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.exceptions import TableQueryValidationError
from app.db.repository import PersistedTableRecord
from app.models.documents import DocumentStatus
from app.models.structured import (
    TableQueryOperation,
    TableQueryPlan,
    TableQueryRequest,
)
from app.services.tables.query import (
    TABLE_PLAN_SYSTEM_PROMPT,
    TableQueryService,
    build_table_plan_prompt,
)


DOCUMENT_ID = uuid4()
TABLE_ID = uuid4()
HEADERS = ["Product", "Revenue", "Units"]
ROWS = [["A", "1200", "10"], ["B", "1750", "8"], ["C", "950", "12"]]


def make_table(
    *,
    headers: list[str] = HEADERS,
    rows: list[list[str]] = ROWS,
) -> PersistedTableRecord:
    """Create one detached persisted-table projection."""

    return PersistedTableRecord(
        table_id=TABLE_ID,
        document_id=DOCUMENT_ID,
        original_filename="sales.pdf",
        page_number=2,
        table_index=1,
        headers=headers,
        rows=rows,
    )


class TableRepository:
    """Repository double for one ready indexed document/table."""

    def __init__(self, table: PersistedTableRecord | None = None) -> None:
        self.document = SimpleNamespace(
            id=DOCUMENT_ID,
            original_filename="sales.pdf",
            status=DocumentStatus.READY,
            is_indexed=True,
        )
        self.table = table or make_table()

    def list_document_tables(self, document_id):
        assert document_id == DOCUMENT_ID
        return self.document, [self.table]


class FixedPlanProvider:
    """Provider double for constrained natural-language plan generation."""

    model = "test-model"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0

    async def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        return self.response


def service(provider: FixedPlanProvider | None = None, table: PersistedTableRecord | None = None) -> TableQueryService:
    """Build a table service with a deterministic provider double."""

    return TableQueryService(
        TableRepository(table),
        provider or FixedPlanProvider({}),
        Settings(table_query_max_rows=100),
    )


def plan(operation: str, **updates: object) -> TableQueryPlan:
    """Build an explicit safe plan for deterministic executor tests."""

    return TableQueryPlan(operation=operation, **updates)


def run_query(query_plan: TableQueryPlan):
    """Run one explicit plan through the service."""

    return asyncio.run(
        service().query(
            DOCUMENT_ID,
            TABLE_ID,
            TableQueryRequest(question="test", plan=query_plan),
        )
    )


def test_inventory_and_preview_preserve_table_identity_dimensions_and_rows() -> None:
    result = service().inventory(DOCUMENT_ID)
    preview = service().preview(DOCUMENT_ID, TABLE_ID, preview_rows=2)

    assert result.tables[0].table_id == TABLE_ID
    assert result.tables[0].filename == "sales.pdf"
    assert result.tables[0].page_number == 2
    assert result.tables[0].row_count == 3
    assert result.tables[0].column_count == 3
    assert preview.rows == ROWS[:2]
    assert preview.truncated is True


@pytest.mark.parametrize(
    ("operation", "column", "expected_product", "expected_value"),
    [
        ("max", "Revenue", "B", "1750"),
        ("min", "Revenue", "C", "950"),
        ("max", "Units", "C", "12"),
    ],
)
def test_min_max_queries_are_deterministic(operation, column, expected_product, expected_value) -> None:
    response = run_query(plan(operation, target_column=column))

    assert response.result.rows[0]["Product"] == expected_product
    assert response.result.rows[0][column] == expected_value
    assert response.sources[0].source_id == "T1"
    assert response.sources[0].row_indices
    assert response.answer.endswith("[T1]")


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("sum", "3900"), ("average", "1300")],
)
def test_numeric_aggregations_preserve_structured_value(operation, expected) -> None:
    response = run_query(plan(operation, target_column="Revenue"))

    assert response.result.value == expected
    assert response.result.rows == []
    assert response.result.row_count == 3


def test_count_filter_and_top_n_operations() -> None:
    count = run_query(plan("count"))
    filtered = run_query(
        plan("filter", filter_column="Revenue", filter_operator="gt", filter_value=1000)
    )
    top = run_query(plan("top_n", target_column="Revenue", limit=2))

    assert count.result.value == 3
    assert [row["Product"] for row in filtered.result.rows] == ["A", "B"]
    assert [row["Product"] for row in top.result.rows] == ["B", "A"]


def test_sort_and_select_preserve_original_display_values() -> None:
    sorted_response = run_query(plan("sort", target_column="Revenue", sort_direction="asc"))
    selected = run_query(plan("select", return_columns=["Product", "Revenue"], limit=2))

    assert [row["Product"] for row in sorted_response.result.rows] == ["C", "A", "B"]
    assert selected.result.rows == [
        {"Product": "A", "Revenue": "1200"},
        {"Product": "B", "Revenue": "1750"},
    ]


def test_currency_values_are_numeric_for_aggregation_but_original_rows_remain_text() -> None:
    currency_table = make_table(
        headers=["Product", "Revenue"],
        rows=[["A", "€1,200.50"], ["B", "€800.50"]],
    )
    response = service(table=currency_table).query
    result = asyncio.run(
        response(
            DOCUMENT_ID,
            TABLE_ID,
            TableQueryRequest(question="total", plan=plan("sum", target_column="Revenue")),
        )
    )

    assert result.result.value == "2001"


@pytest.mark.parametrize(
    "query_plan",
    [
        plan("max", target_column="Missing"),
        plan("sum", target_column="Product"),
        plan("filter", filter_column="Missing", filter_operator="eq", filter_value="A"),
    ],
)
def test_invalid_column_and_nonnumeric_aggregation_are_rejected(query_plan: TableQueryPlan) -> None:
    with pytest.raises(TableQueryValidationError):
        run_query(query_plan)


def test_malicious_table_cell_is_data_and_never_executed() -> None:
    malicious = make_table(
        headers=["Product", "Revenue"],
        rows=[["DROP TABLE documents;", "10"]],
    )
    response = asyncio.run(
        service(table=malicious).query(
            DOCUMENT_ID,
            TABLE_ID,
            TableQueryRequest(question="show rows", plan=plan("select")),
        )
    )

    assert response.result.rows == [{"Product": "DROP TABLE documents;", "Revenue": "10"}]


def test_natural_language_plan_uses_mocked_ollama_and_validates_columns() -> None:
    provider = FixedPlanProvider(
        {
            "operation": "max",
            "target_column": "Revenue",
            "return_columns": ["Product", "Revenue"],
            "filter_column": None,
            "filter_operator": None,
            "filter_value": None,
            "sort_direction": "desc",
            "limit": 10,
        }
    )
    result = asyncio.run(
        TableQueryService(TableRepository(), provider, Settings()).query(
            DOCUMENT_ID,
            TABLE_ID,
            TableQueryRequest(question="Show the revenue leader by product."),
        )
    )

    assert provider.calls == 1
    assert result.plan.operation is TableQueryOperation.MAX
    assert result.result.rows[0]["Product"] == "B"
    assert result.plan_generation_time_ms >= 0


def test_plan_prompt_and_system_prompt_forbid_sql_python_and_cell_instructions() -> None:
    prompt = build_table_plan_prompt("Which product is highest?", HEADERS)

    assert "Revenue" in prompt
    assert "SQL" in TABLE_PLAN_SYSTEM_PROMPT
    assert "Python" in TABLE_PLAN_SYSTEM_PROMPT
    assert "executable" in TABLE_PLAN_SYSTEM_PROMPT


def test_unknown_operation_and_extra_plan_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        TableQueryPlan(operation="python_eval")
    with pytest.raises(ValueError):
        TableQueryPlan.model_validate({"operation": "count", "code": "DROP TABLE"})
