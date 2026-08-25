"""Focused tests for the Module 13 presentation helpers and E5 read-only view."""

from __future__ import annotations

import importlib

import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app.api.client import ApiError
from streamlit_app.app import PAGES
from streamlit_app.evaluation import (
    E5UiArtifactError,
    find_scorecard_metric,
    format_e5_metric,
    load_e5_package,
    limitation_display_title,
    public_provenance,
    retrieval_rows,
)
from streamlit_app.ui_helpers import (
    conversation_display_labels,
    friendly_api_error_message,
    page_display_data,
    prepare_source_rows,
    set_selected_document,
    upload_size_error,
)


def test_e5_package_uses_authoritative_run_and_preserves_metric_status() -> None:
    package = load_e5_package()

    assert package["summary"]["run_id"] == "final_baseline_20260821_final"
    assert [row["method"] for row in retrieval_rows(package)] == [
        "keyword",
        "semantic",
        "hybrid",
        "hybrid_reranked",
    ]
    record = find_scorecard_metric(package, "layout.f1", module="E2", dataset="doclaynet")
    assert record is not None
    assert record["status"] == "MEASURED"
    assert format_e5_metric(record, percent=True) == "16.970%"


def test_e5_metric_formatting_does_not_turn_blocked_into_zero() -> None:
    assert format_e5_metric({"status": "BLOCKED", "value": None}) == "BLOCKED"
    assert format_e5_metric(None) == "—"


def test_public_evaluation_view_excludes_portfolio_content_and_absolute_paths() -> None:
    package = load_e5_package()
    provenance = public_provenance(package)

    assert "portfolio_metrics" not in package
    assert provenance == {
        "Run": "final_baseline_20260821_final",
        "Datasets": "FUNSD, DocLayNet, DocVQA",
        "DocVQA split": "validation",
        "Manifest": "evaluation/e5/baseline_manifest.json",
    }
    assert not any("D:" in value or "\\" in value for value in provenance.values())


def test_evaluation_limitations_use_public_titles_without_internal_identifiers() -> None:
    package = load_e5_package()
    titles = [limitation_display_title(item.get("id")) for item in package["limitations"]]

    assert titles == [
        "Bounded evaluation sample",
        "Document processing coverage",
        "OCR quality",
        "Layout quality",
        "Answer indexability",
        "Document size limit",
        "Retrieval evaluation scope",
        "Reranking latency",
        "Local LLM runtime",
        "Answer-format evaluation",
        "Citation sample size",
        "Extended runtime diagnostic",
        "Metric interpretation",
    ]
    assert not any(
        forbidden in title
        for title in titles
        for forbidden in ("CONFIRMED", "METHODOLOGICAL_RULE", "bounded_e2", "doclaynet_failure", "e41_blocked")
    )


def _render_evaluation_ui_for_app_test() -> None:
    from streamlit_app.evaluation import render_evaluation

    render_evaluation()


def test_evaluation_render_path_uses_public_limitation_headings() -> None:
    app_test = AppTest.from_function(_render_evaluation_ui_for_app_test)
    app_test.run()

    assert not app_test.exception
    expander_labels = {item.label for item in app_test.expander}
    assert "Bounded evaluation sample" in expander_labels
    assert "Metric interpretation" in expander_labels
    assert not any(
        forbidden in label
        for label in expander_labels
        for forbidden in ("CONFIRMED", "METHODOLOGICAL_RULE", "bounded_e2", "doclaynet_failure", "e41_blocked")
    )


def _render_analyze_ui_for_app_test() -> None:
    """Render Analyze with deterministic HTTP-adapter doubles for UI regression coverage."""

    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            if path == "/api/v1/documents/doc-1":
                return {
                    "id": "doc-1",
                    "original_filename": "manual_module5_employment.pdf",
                    "page_count": 2,
                    "chunk_count": 3,
                    "is_indexed": True,
                }
            if path == "/api/v1/documents/doc-1/tables":
                return {"tables": []}
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.documents = [
        {
            "id": "doc-1",
            "original_filename": "manual_module5_employment.pdf",
            "status": "ready",
            "is_indexed": True,
        }
    ]
    app.render_analyze(DummyClient())


def test_analyze_render_path_exposes_all_four_tabs_and_structured_controls() -> None:
    app_test = AppTest.from_function(_render_analyze_ui_for_app_test)
    app_test.run()

    assert not app_test.exception
    assert {item.label for item in app_test.tabs} >= {"Summary", "Classification", "Extraction", "Tables"}
    assert {item.value for item in app_test.subheader} >= {
        "Summarize document",
        "Classify document",
        "Structured extraction",
        "Table query",
    }
    assert any(item.label == "Requested fields" for item in app_test.text_area)
    assert any(item.label == "Run extraction" for item in app_test.button)
    assert any(
        item.value == "No structured tables were detected for this document."
        for item in app_test.info
    )


def test_upload_size_validation_allows_exact_25_mb_boundary() -> None:
    exact_size = 25 * 1024 * 1024

    assert upload_size_error("exact.pdf", exact_size, 25) is None
    assert "25 MB" in (upload_size_error("large.pdf", exact_size + 1, 25) or "")


def test_conversation_labels_are_readable_and_hide_uuid() -> None:
    conversation_id = "b0d729e4-be25-4777-8dde-43899b147a0a"
    labels = conversation_display_labels(
        [
            {
                "id": conversation_id,
                "title": "Notice policy",
                "created_at": "2026-08-22T10:30:00+00:00",
            },
            {
                "id": "another-id",
                "title": "Notice policy",
                "created_at": "2026-08-22T11:30:00+00:00",
            },
        ]
    )

    assert labels[conversation_id] == "Notice policy · 2026-08-22 10:30"
    assert conversation_id not in labels[conversation_id]
    assert "another-id" not in labels["another-id"]


def test_ocr_page_display_is_collapsed_and_does_not_clean_text() -> None:
    display = page_display_data(
        {
            "extraction_method": "ocr",
            "ocr_applied": True,
            "text": "NOISY OCR $$$ text",
        }
    )

    assert display["is_ocr"] is True
    assert display["extraction_method"] == "OCR"
    assert display["expander_label"] == "View extracted OCR text"
    assert display["text"] == "NOISY OCR $$$ text"


def test_missing_e5_artifact_is_a_controlled_error(tmp_path) -> None:
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(E5UiArtifactError, match="authoritative E5 evaluation package"):
        load_e5_package(tmp_path)


def test_document_selection_clears_results_from_previous_document() -> None:
    state = {
        "selected_document_id": "old",
        "last_search": {"results": ["old"]},
        "last_rag": {"answer": "old"},
        "last_summary": {"summary": "old"},
        "last_classification": {"selected_label": "old"},
        "last_extraction": {"fields": []},
        "last_table_query": {"answer": "old"},
        "last_comparison": {"changes": []},
        "last_pii_scan": {"detections": []},
        "last_pii_redaction": {"artifact": "old"},
    }

    set_selected_document(state, "new")

    assert state["selected_document_id"] == "new"
    assert all(state[key] is None for key in state if key.startswith("last_"))


def test_source_rows_preserve_metadata_without_inventing_pages() -> None:
    rows = prepare_source_rows(
        [
            {
                "source_id": "S1",
                "filename": "policy.pdf",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "start_page": 4,
                "end_page": 5,
                "excerpt": "Cancellation requires notice.",
                "final_rank": 1,
            },
            {"source_id": "S2", "filename": "other.pdf", "excerpt": "No page metadata."},
        ]
    )

    assert rows[0]["Page"] == "4–5"
    assert rows[0]["Document ID"] == "doc-1"
    assert "Page" not in rows[1]
    assert rows[1]["Excerpt"] == "No page metadata."


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ApiError("The DocuIntel API is unavailable."), "local DocuIntel service is unavailable"),
        (ApiError("Invalid field", status_code=422), "Please check the submitted values"),
        (ApiError("Not found", status_code=404), "resource was not found"),
    ],
)
def test_api_errors_are_safe_presentation_copy(error: ApiError, expected: str) -> None:
    assert expected in friendly_api_error_message(error)


def test_navigation_groups_ask_workflows_and_adds_evaluation() -> None:
    assert PAGES == ("Home", "Documents", "Ask", "Analyze", "Compare", "Privacy", "Evaluation")


def test_production_module13_import_contract_is_not_mocked() -> None:
    """The real app and status modules must expose every Module 13 import."""

    status_module = importlib.import_module("streamlit_app.components.status")
    evaluation_module = importlib.import_module("streamlit_app.evaluation")
    helpers_module = importlib.import_module("streamlit_app.ui_helpers")
    app_module = importlib.import_module("streamlit_app.app")

    assert callable(status_module.render_system_status)
    assert app_module.render_system_status is status_module.render_system_status
    assert app_module.render_evaluation is evaluation_module.render_evaluation
    assert callable(helpers_module.friendly_api_error_message)
    assert callable(helpers_module.set_selected_document)
