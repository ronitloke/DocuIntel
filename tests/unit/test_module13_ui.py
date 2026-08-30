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
    set_comparison_pair,
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
    # The full suite loads the embedding and reranking models before this UI
    # test; Streamlit's default three-second AppTest timeout is too tight on
    # Windows under that workload. Keep the same assertions with a bounded,
    # platform-neutral timeout.
    app_test.run(timeout=10)

    assert not app_test.exception
    rendered = "\n".join(item.value for item in app_test.markdown)
    assert "MODEL &amp; SYSTEM EVALUATION" in rendered
    assert "AUTHORITATIVE BENCHMARK SNAPSHOT" in rendered
    assert "Documents prepared" in rendered and ">25<" in rendered
    assert "Documents indexed" in rendered and ">24<" in rendered
    assert "Scorable questions" in rendered and ">43<" in rendered
    assert "Measured records" in rendered and ">133<" in rendered
    assert "Measured" in rendered and ">133<" in rendered
    assert "Blocked" in rendered and ">3<" in rendered
    assert "Not applicable" in rendered and ">9<" in rendered
    assert {item.label for item in app_test.tabs} >= {
        "Document understanding",
        "Retrieval",
        "RAG reliability",
        "Limitations & provenance",
    }
    expander_labels = {item.label for item in app_test.expander}
    assert "Bounded evaluation sample" in expander_labels
    assert "Metric interpretation" in expander_labels
    assert not any(
        forbidden in label
        for label in expander_labels
        for forbidden in ("CONFIRMED", "METHODOLOGICAL_RULE", "bounded_e2", "doclaynet_failure", "e41_blocked")
    )
    forbidden_ui = (
        "portfolio",
        "resume",
        "API Online",
        "API version",
        "Database Ready",
        "System Status",
        "Refresh backend status",
        "D:\\",
        "C:\\Users\\",
    )
    assert not any(term.casefold() in rendered.casefold() for term in forbidden_ui)


def _render_e4_count_metrics_for_app_test() -> None:
    from streamlit_app.evaluation import (
        _metric_card,
        find_scorecard_metric,
        load_e5_package,
    )

    package = load_e5_package()
    for method in ("hybrid", "hybrid_reranked"):
        for metric, label in (
            ("questions_answered", "Questions answered"),
            ("questions_failed", "Questions failed"),
            ("failure_ollama_timeout", "Ollama timeouts"),
            ("failure_no_literal_answer_match", "No literal answer match"),
        ):
            record = find_scorecard_metric(package, f"{method}.{metric}", module="E4", dataset="docvqa")
            _metric_card(record, f"{method} · {label}", decimals=0)


def test_e4_count_metrics_render_as_integers() -> None:
    app_test = AppTest.from_function(_render_e4_count_metrics_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    values_by_label = {item.label: item.value for item in app_test.metric}
    assert values_by_label == {
        "hybrid · Questions answered": "3",
        "hybrid · Questions failed": "97",
        "hybrid · Ollama timeouts": "40",
        "hybrid · No literal answer match": "57",
        "hybrid_reranked · Questions answered": "6",
        "hybrid_reranked · Questions failed": "94",
        "hybrid_reranked · Ollama timeouts": "37",
        "hybrid_reranked · No literal answer match": "57",
    }


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
    markdown_values = [item.value for item in app_test.markdown]
    assert any("DOCUMENT INTELLIGENCE WORKSPACE" in value for value in markdown_values)
    assert any("Analyze" in value for value in markdown_values)
    assert any("grounded evidence" in value for value in markdown_values)
    assert any("Document summary" in value for value in markdown_values)
    assert any("Document classification" in value for value in markdown_values)
    assert any("Structured extraction" in value for value in markdown_values)
    assert any("Table intelligence" in value for value in markdown_values)
    assert any(item.label == "Requested fields" for item in app_test.text_area)
    assert any(item.label == "Run extraction" for item in app_test.button)
    assert any(
        item.value == "No structured tables were detected for this document."
        for item in app_test.info
    )
    forbidden = (
        "API Online",
        "API version",
        "Database Ready",
        "System Status",
        "Refresh backend status",
        "Built with FastAPI",
    )
    assert not any(term in "\n".join(markdown_values) for term in forbidden)


def _render_analyze_multiple_documents_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    documents = [
        {
            "id": "doc-vqa",
            "original_filename": "docvqa-example.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "doc-manual",
            "original_filename": "manual_module5_employment.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "doc-layout",
            "original_filename": "layout_table_module4.pdf",
            "status": "ready",
            "is_indexed": True,
        },
    ]

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            if path == "/api/v1/documents":
                return {"items": documents}
            document_id = path.removeprefix("/api/v1/documents/")
            if document_id in {item["id"] for item in documents}:
                item = next(item for item in documents if item["id"] == document_id)
                return {
                    **item,
                    "page_count": 1,
                    "chunk_count": 1,
                }
            if path.endswith("/tables"):
                return {"tables": []}
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.documents = None
    app.render_analyze(DummyClient())


def test_analyze_selector_uses_complete_api_document_list() -> None:
    app_test = AppTest.from_function(_render_analyze_multiple_documents_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    selector = app_test.selectbox[0]
    assert list(selector.options) == [
        "docvqa-example.pdf · ready",
        "manual_module5_employment.pdf · ready",
        "layout_table_module4.pdf · ready",
    ]
    assert selector.format_func("doc-manual") == "manual_module5_employment.pdf · ready"
    assert selector.format_func("doc-layout") == "layout_table_module4.pdf · ready"

    selector.select("doc-manual")
    app_test.run(timeout=10)

    assert not app_test.exception
    assert app_test.selectbox[0].value == "doc-manual"
    assert any("manual_module5_employment.pdf" in item.value for item in app_test.markdown)


def _render_home_count_cache_contract_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents"
            return {
                "items": [{"id": "first-page-only", "original_filename": "first.pdf"}],
                "total": 32,
            }

    app.initialize_state()
    st.session_state.documents = None
    st.session_state.home_document_count = None
    st.session_state.home_document_count_loaded = False
    count = app._load_home_document_count(DummyClient())
    st.write(f"count={count}; documents_is_none={st.session_state.documents is None}")


def test_home_count_does_not_replace_full_document_cache_with_page_one() -> None:
    app_test = AppTest.from_function(_render_home_count_cache_contract_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    assert any("count=32; documents_is_none=True" in item.value for item in app_test.markdown)


def _render_analyze_long_filename_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    long_filename = "annual_" + ("very-long-document-name-" * 12) + "<policy>.pdf"

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            if path == "/api/v1/documents/doc-long":
                return {
                    "id": "doc-long",
                    "original_filename": long_filename,
                    "status": "ready",
                    "page_count": 4,
                    "chunk_count": 8,
                    "is_indexed": True,
                }
            if path == "/api/v1/documents/doc-long/tables":
                return {"tables": []}
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.documents = [
        {
            "id": "doc-long",
            "original_filename": long_filename,
            "status": "ready",
            "is_indexed": True,
        }
    ]
    app.render_analyze(DummyClient())


def test_analyze_long_filename_is_escaped_without_breaking_markup() -> None:
    app_test = AppTest.from_function(_render_analyze_long_filename_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("&lt;policy&gt;.pdf" in value for value in markdown_values)


def _render_analyze_state_isolation_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            if path == "/api/v1/documents/doc-new":
                return {
                    "id": "doc-new",
                    "original_filename": "new-policy.pdf",
                    "status": "ready",
                    "page_count": 1,
                    "chunk_count": 1,
                    "is_indexed": True,
                }
            if path == "/api/v1/documents/doc-new/tables":
                return {"tables": []}
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.documents = [
        {
            "id": "doc-new",
            "original_filename": "new-policy.pdf",
            "status": "ready",
            "is_indexed": True,
        }
    ]
    st.session_state.selected_document_id = "doc-old"
    st.session_state.last_summary = {
        "document_id": "doc-old",
        "summary": "STALE SUMMARY MUST NOT RENDER",
    }
    st.session_state.last_classification = {
        "document_id": "doc-old",
        "selected_label": "STALE CLASSIFICATION MUST NOT RENDER",
    }
    st.session_state.last_extraction = {
        "document_id": "doc-old",
        "fields": [{"field": "stale", "status": "found", "value": "STALE EXTRACTION MUST NOT RENDER"}],
    }
    st.session_state.last_table_query = {
        "document_id": "doc-old",
        "table": {"table_id": "old-table"},
        "answer": "STALE TABLE RESULT MUST NOT RENDER",
    }
    app.render_analyze(DummyClient())


def test_analyze_document_change_does_not_render_previous_document_results() -> None:
    app_test = AppTest.from_function(_render_analyze_state_isolation_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    rendered = "\n".join(item.value for item in app_test.markdown)
    assert "STALE SUMMARY MUST NOT RENDER" not in rendered
    assert "STALE CLASSIFICATION MUST NOT RENDER" not in rendered
    assert "STALE EXTRACTION MUST NOT RENDER" not in rendered
    assert "STALE TABLE RESULT MUST NOT RENDER" not in rendered


def _render_compare_form_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    documents = [
        {
            "id": "base-id",
            "original_filename": "module12_3_base.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "target-id",
            "original_filename": "module12_3_target.pdf",
            "status": "ready",
            "is_indexed": True,
        },
    ]

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents"
            return {"items": documents}

    app.initialize_state()
    st.session_state.documents = None
    app.render_compare(DummyClient())


def test_compare_workspace_preserves_all_controls_and_document_pair_options() -> None:
    app_test = AppTest.from_function(_render_compare_form_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    assert any("DOCUMENT CHANGE INTELLIGENCE" in item.value for item in app_test.markdown)
    assert {item.label for item in app_test.selectbox} >= {
        "Base document",
        "Target document",
        "Comparison mode",
    }
    assert all(
        set(item.options) == {"module12_3_base.pdf · ready", "module12_3_target.pdf · ready"}
        for item in app_test.selectbox[:2]
    )
    assert {item.label for item in app_test.checkbox} >= {
        "Compare structured tables",
        "Include unchanged items",
        "Generate grounded change summary",
    }
    assert any(item.label == "Compare documents" for item in app_test.button)
    assert not any(
        term in "\n".join(item.value for item in app_test.markdown)
        for term in (
            "API Online",
            "API version",
            "Database Ready",
            "System Status",
            "Refresh backend status",
            "Built with FastAPI",
        )
    )


def _render_compare_result_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    documents = [
        {
            "id": "base-id",
            "original_filename": "module12_3_base.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "target-id",
            "original_filename": "module12_3_target.pdf",
            "status": "ready",
            "is_indexed": True,
        },
    ]
    response = {
        "base_document": {"document_id": "base-id", "filename": "module12_3_base.pdf"},
        "target_document": {"document_id": "target-id", "filename": "module12_3_target.pdf"},
        "statistics": {
            "added_count": 2,
            "removed_count": 1,
            "modified_count": 4,
            "unchanged_count": 1,
            "table_change_count": 3,
        },
        "summary": "Changed from thirty to forty-five days [A1][B1].",
        "changes": [
            {
                "change_id": "C1",
                "change_type": "modified",
                "scope": "text",
                "base_text": "Thirty days written notice.",
                "target_text": "Forty-five days written notice.",
                "base_provenance": [
                    {
                        "source_id": "A1",
                        "filename": "module12_3_base.pdf",
                        "page_number": 1,
                        "section_heading": "Employment Policy",
                    }
                ],
                "target_provenance": [
                    {
                        "source_id": "B1",
                        "filename": "module12_3_target.pdf",
                        "page_number": 1,
                        "section_heading": "Employment Policy",
                    }
                ],
            },
            {
                "change_id": "C2",
                "change_type": "added",
                "scope": "table",
                "base_text": None,
                "target_text": "Product: Keyboard | Qty: 2 | Price: 80",
                "base_provenance": [],
                "target_provenance": [
                    {
                        "source_id": "B100001",
                        "filename": "module12_3_target.pdf",
                        "page_number": 4,
                    }
                ],
                "table_detail": {
                    "table_change_type": "row_added",
                    "row_key": "keyboard",
                    "column": None,
                    "before": None,
                    "after": None,
                    "row_values": {"Product": "Keyboard", "Qty": "2", "Price": "80"},
                },
            },
            {
                "change_id": "C3",
                "change_type": "removed",
                "scope": "text",
                "base_text": "Annual training is mandatory.",
                "target_text": None,
                "base_provenance": [
                    {
                        "source_id": "A2",
                        "filename": "module12_3_base.pdf",
                        "page_number": 2,
                    }
                ],
                "target_provenance": [],
            },
            {
                "change_id": "C4",
                "change_type": "unchanged",
                "scope": "table",
                "base_text": "Product: Mouse | Qty: 5 | Price: 20",
                "target_text": "Product: Mouse | Qty: 5 | Price: 20",
                "base_provenance": [{"source_id": "A100001", "filename": "module12_3_base.pdf", "page_number": 4}],
                "target_provenance": [{"source_id": "B100001", "filename": "module12_3_target.pdf", "page_number": 4}],
                "table_detail": {
                    "table_change_type": "table_unchanged",
                    "row_key": "mouse",
                    "column": None,
                    "before": None,
                    "after": None,
                    "row_values": {},
                },
            },
        ],
        "total_time_ms": 12.5,
    }

    class DummyClient:
        pass

    app.initialize_state()
    st.session_state.documents = documents
    st.session_state["comparison-base-document"] = "base-id"
    st.session_state["comparison-target-document"] = "target-id"
    st.session_state.comparison_pair = ("base-id", "target-id")
    st.session_state.last_comparison = response
    app.render_compare(DummyClient())


def test_compare_result_preserves_counts_sources_table_details_and_content() -> None:
    app_test = AppTest.from_function(_render_compare_result_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    rendered = "\n".join(markdown_values)
    assert "COMPARISON RESULT" in rendered
    assert "module12_3_base.pdf" in rendered
    assert "module12_3_target.pdf" in rendered
    assert "CHANGE SUMMARY" in rendered
    assert "Changed from thirty to forty-five days" in rendered
    assert all(str(value) in rendered for value in (2, 1, 4, 3))
    assert {item.label for item in app_test.expander} >= {
        "C1 · Text",
        "C2 · Table",
        "C3 · Text",
        "C4 · Table",
    }
    assert all(term in rendered for term in ("BASE CONTENT", "TARGET CONTENT", "A1", "B1", "Column", "Row / key"))
    assert "API Online" not in rendered
    assert "System Status" not in rendered


def test_comparison_pair_change_clears_previous_result() -> None:
    state = {
        "comparison_pair": ("base-id", "target-id"),
        "last_comparison": {"statistics": {"modified_count": 1}},
    }

    set_comparison_pair(state, "base-id", "different-target")

    assert state["comparison_pair"] == ("base-id", "different-target")
    assert state["last_comparison"] is None


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


def _render_ask_page_ui_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object] | list[object]:
            if path == "/api/v1/documents":
                return {
                    "items": [
                        {
                            "id": "doc-1",
                            "original_filename": "employee-policy.pdf",
                            "status": "ready",
                            "is_indexed": True,
                        }
                    ]
                }
            if path == "/api/v1/conversations":
                return []
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.documents = None
    st.session_state.conversations = None
    app.render_ask(DummyClient())


def test_ask_page_renders_all_workspaces_and_preserves_removed_status_controls() -> None:
    app_test = AppTest.from_function(_render_ask_page_ui_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    assert {item.label for item in app_test.tabs} >= {
        "Grounded Q&A",
        "Search evidence",
        "Conversations",
    }
    assert {item.label for item in app_test.button} >= {
        "Ask",
        "Search",
        "New conversation",
    }
    assert any(item.label == "Question" for item in app_test.text_area)
    assert any(item.label == "Search query" for item in app_test.text_input)
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Ask AI" in value for value in markdown_values)
    assert any("Search document evidence" in value for value in markdown_values)
    assert any("No conversations yet" in item.value for item in app_test.info)
    forbidden = (
        "API Online",
        "API version",
        "Database Ready",
        "System Status",
        "Refresh backend status",
        "Built with FastAPI",
    )
    assert not any(term in "\n".join(markdown_values) for term in forbidden)
    assert not any(term in item.label for item in app_test.button for term in forbidden)


def _render_grounded_answer_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents"
            return {"items": []}

    app.initialize_state()
    st.session_state.documents = None
    st.session_state.last_rag = {
        "answer": "The cancellation period is thirty days. [S1]",
        "citations_valid": True,
        "sources": [
            {
                "source_id": "S1",
                "filename": "employee-policy.pdf",
                "start_page": 4,
                "end_page": 4,
                "excerpt": "Employees may cancel with thirty days notice.",
                "final_rank": 1,
                "chunk_id": "chunk-1",
                "reranker_score": 7.2,
            }
        ],
        "retrieval_time_ms": 12.5,
    }
    app.render_grounded_ask(DummyClient())


def test_ask_answer_and_source_rendering_preserve_content_and_metadata() -> None:
    app_test = AppTest.from_function(_render_grounded_answer_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Grounded answer" in value for value in markdown_values)
    assert any("The cancellation period is thirty days." in value for value in markdown_values)
    assert any("Evidence returned by the grounded retrieval pipeline." in value for value in markdown_values)
    assert "S1 · employee-policy.pdf · page 4" in {item.label for item in app_test.expander}
    assert any("Employees may cancel with thirty days notice." in value for value in markdown_values)
    assert any("Final rank" in value and "Reranker score" in value for value in markdown_values)


def _render_search_results_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents"
            return {"items": []}

    app.initialize_state()
    st.session_state.documents = None
    st.session_state.last_search = {
        "results": [
            {
                "rank": 1,
                "original_filename": "employee-policy.pdf",
                "start_page": 4,
                "end_page": 4,
                "text": "Cancellation requires thirty days notice.",
                "retrieval_method": "hybrid",
                "base_rank": 1,
                "content_type": "text",
                "rerank_score": 5.5,
            }
        ]
    }
    app.render_search(DummyClient(), nested=True)


def test_search_evidence_rendering_preserves_result_details() -> None:
    app_test = AppTest.from_function(_render_search_results_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Search document evidence" in value for value in markdown_values)
    assert any("Cancellation requires thirty days notice." in value for value in markdown_values)
    assert any("Method" in value and "hybrid" in value for value in markdown_values)
    assert "#1 · employee-policy.pdf · page 4" in {item.label for item in app_test.expander}


def _render_conversation_ui_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> list[object] | dict[str, object]:
            if path == "/api/v1/conversations":
                return [
                    {
                        "id": "conversation-1",
                        "title": "Policy review",
                        "created_at": "2026-08-22T10:30:00+00:00",
                    }
                ]
            if path == "/api/v1/conversations/conversation-1/messages":
                return [
                    {"role": "user", "content": "What is the cancellation period?"},
                    {"role": "assistant", "content": "Thirty days. [S1]"},
                ]
            if path == "/api/v1/documents":
                return {"items": []}
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.conversations = None
    st.session_state.documents = None
    app.render_conversations(DummyClient(), nested=True)


def test_conversation_workspace_renders_history_filters_and_delete_confirmation() -> None:
    app_test = AppTest.from_function(_render_conversation_ui_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Continue persistent document-grounded conversations" in value for value in markdown_values)
    assert any("USER" in value for value in markdown_values)
    assert any("DOCUINTEL" in value for value in markdown_values)
    assert any("What is the cancellation period?" in value for value in markdown_values)
    assert any(item.label == "Delete conversation" for item in app_test.button)
    assert any(item.label == "I understand this deletes the conversation" for item in app_test.checkbox)


def _render_home_ui_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents"
            return {"items": [], "total": 7}

    app.initialize_state()
    st.session_state.backend_status = {
        "health": {"status": "healthy", "version": "0.1.0"},
        "ready": {"status": "ready"},
    }
    st.session_state.home_document_count = None
    st.session_state.home_document_count_loaded = False
    app.render_home(DummyClient())


def test_home_redesign_renders_hero_actions_kpis_and_capabilities() -> None:
    app_test = AppTest.from_function(_render_home_ui_for_app_test)
    app_test.run()

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Welcome back" in value for value in markdown_values)
    assert any("Turn documents into" in value for value in markdown_values)
    assert any("actionable" in value and "intelligence." in value for value in markdown_values)
    assert any("Document Intelligence" in value for value in markdown_values)
    assert any("Grounded RAG" in value for value in markdown_values)
    assert any(">7<" in value for value in markdown_values)
    assert {item.label for item in app_test.button} >= {"Upload Documents", "Ask DocuIntel"}
    assert "Refresh backend status" not in {item.label for item in app_test.button}
    assert not any("System status" in value for value in markdown_values)


def _render_documents_empty_ui_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents"
            return {"items": []}

    app.initialize_state()
    st.session_state.documents = None
    app.render_documents(DummyClient())


def test_documents_page_preserves_multi_upload_and_empty_library_state() -> None:
    app_test = AppTest.from_function(_render_documents_empty_ui_for_app_test)
    app_test.run()

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Documents" in value for value in markdown_values)
    assert any("Upload documents" in value for value in markdown_values)
    assert any("Document library" in value for value in markdown_values)
    assert any(item.label == "Upload selected PDFs" for item in app_test.button)
    assert len(app_test.file_uploader) == 1
    assert app_test.file_uploader[0].accept_multiple_files is True
    assert any(item.value == "No documents have been uploaded yet." for item in app_test.info)


def _render_selected_documents_ui_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            if path == "/api/v1/documents":
                return {
                    "items": [
                        {
                            "id": "doc-1",
                            "original_filename": "employee-policy.pdf",
                            "status": "ready",
                            "is_indexed": True,
                        }
                    ]
                }
            if path == "/api/v1/documents/doc-1":
                return {
                    "id": "doc-1",
                    "original_filename": "employee-policy.pdf",
                    "status": "ready",
                    "page_count": 1,
                    "chunk_count": 1,
                    "is_indexed": True,
                    "metadata": {"title": "Employee policy"},
                    "summary": {"extraction_method": "native"},
                }
            if path == "/api/v1/documents/doc-1/pages":
                return {"items": [{"page_number": 1}]}
            if path == "/api/v1/documents/doc-1/pages/1":
                return {
                    "page_number": 1,
                    "extraction_method": "native",
                    "text": "Native extracted policy text.",
                    "ocr_applied": False,
                }
            if path == "/api/v1/documents/doc-1/chunks":
                return {
                    "items": [
                        {
                            "id": "chunk-1",
                            "sequence_number": 1,
                            "start_page": 1,
                            "end_page": 1,
                            "content_type": "paragraph",
                            "character_count": 31,
                            "text": "Native extracted policy text.",
                        }
                    ]
                }
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    st.session_state.documents = None
    app.render_documents(DummyClient())


def test_documents_page_renders_selected_overview_metadata_pages_and_chunks() -> None:
    app_test = AppTest.from_function(_render_selected_documents_ui_for_app_test)
    app_test.run()

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("employee-policy.pdf" in value for value in markdown_values)
    assert any("SELECTED DOCUMENT" in value for value in markdown_values)
    assert any("Extraction method: native" in value for value in markdown_values)
    assert any("Extracted text" in value for value in markdown_values)
    assert any("Chunk ID: chunk-1" in value for value in markdown_values)
    assert {item.label for item in app_test.tabs} >= {"Pages (1)", "Chunks (1)"}
    assert "Metadata and processing summary" in {item.label for item in app_test.expander}
    assert any(item.label == "Rebuild index" for item in app_test.button)
    assert any(item.label == "Delete document" for item in app_test.button)
    assert any(item.label == "I understand this deletes the document" for item in app_test.checkbox)


def _render_ocr_page_for_app_test() -> None:
    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            assert path == "/api/v1/documents/doc-ocr/pages/1"
            return {
                "page_number": 1,
                "extraction_method": "ocr",
                "ocr_applied": True,
                "text": "Scanned OCR text.",
            }

    app.initialize_state()
    app.render_pages(DummyClient(), "doc-ocr", [{"page_number": 1}])


def test_documents_page_preserves_collapsed_ocr_text_state() -> None:
    app_test = AppTest.from_function(_render_ocr_page_for_app_test)
    app_test.run()

    assert not app_test.exception
    markdown_values = [item.value for item in app_test.markdown]
    assert any("Extraction method: OCR" in value for value in markdown_values)
    assert any(item.value == "OCR quality may vary for scanned documents." for item in app_test.caption)
    assert "View extracted OCR text" in {item.label for item in app_test.expander}


def _render_sidebar_ui_for_app_test() -> None:
    from streamlit_app import app

    class DummyClient:
        def get(self, path: str, **_: object) -> dict[str, object]:
            if path == "/health":
                return {"status": "healthy", "version": "0.1.0"}
            if path == "/ready":
                return {"status": "ready"}
            raise AssertionError(f"Unexpected test API path: {path}")

    app.initialize_state()
    app.render_sidebar(DummyClient())


def test_sidebar_uses_button_navigation_without_footer_or_status_branding() -> None:
    app_test = AppTest.from_function(_render_sidebar_ui_for_app_test)
    app_test.run()

    assert not app_test.exception
    assert {item.label for item in app_test.button} >= {
        "Home",
        "Documents",
        "Ask AI",
        "Analyze",
        "Compare",
        "Privacy",
        "Evaluation",
    }
    assert "Refresh backend status" not in {item.label for item in app_test.button}
    sidebar_markdown = [item.value for item in app_test.markdown]
    assert any('di-sidebar-section-label">MAIN' in value for value in sidebar_markdown)
    assert not any("di-sidebar-footer" in value for value in sidebar_markdown)
    assert not any("Built for local intelligence" in value for value in sidebar_markdown)
    assert not any("Built with FastAPI" in value for value in sidebar_markdown)
    assert not any("SYSTEM" in value for value in sidebar_markdown)
    assert not any("API Online" in value or "Database Ready" in value for value in sidebar_markdown)


def test_production_module13_import_contract_is_not_mocked() -> None:
    """The real app and status modules must expose every Module 13 import."""

    status_module = importlib.import_module("streamlit_app.components.status")
    evaluation_module = importlib.import_module("streamlit_app.evaluation")
    helpers_module = importlib.import_module("streamlit_app.ui_helpers")
    styles_module = importlib.import_module("streamlit_app.styles")
    app_module = importlib.import_module("streamlit_app.app")

    assert callable(status_module.render_system_status)
    assert app_module.render_system_status is status_module.render_system_status
    assert app_module.render_evaluation is evaluation_module.render_evaluation
    for helper_name in (
        "apply_visual_system",
        "render_pipeline_visual",
        "render_sidebar_brand",
    ):
        assert callable(getattr(styles_module, helper_name))
        assert getattr(app_module, helper_name) is getattr(styles_module, helper_name)
    assert not hasattr(styles_module, "render_sidebar_footer_brand")
    assert not hasattr(app_module, "render_sidebar_footer_brand")
    assert callable(helpers_module.friendly_api_error_message)
    assert callable(helpers_module.set_selected_document)


def _render_privacy_form_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    app.initialize_state()
    st.session_state.documents = [
        {
            "id": "doc-pii",
            "original_filename": "module12_4_pii.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "doc-manual",
            "original_filename": "manual_module5_employment.pdf",
            "status": "ready",
            "is_indexed": True,
        },
    ]

    class PrivacyUiClient:
        pass

    app.render_privacy(PrivacyUiClient())


def test_privacy_page_preserves_scan_controls_and_supported_pii_values() -> None:
    app_test = AppTest.from_function(_render_privacy_form_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    assert any(item.label == "Document" for item in app_test.selectbox)
    pii_selector = next(item for item in app_test.multiselect if item.label == "PII types")
    assert set(pii_selector.options) == {"email", "phone_number", "iban", "credit_card"}
    assert any(item.label == "Scan for PII" for item in app_test.button)
    rendered = "\n".join(item.value for item in app_test.markdown)
    assert "PRIVACY &amp; DOCUMENT PROTECTION" in rendered
    assert "Privacy &amp; Redaction" in rendered
    assert "Scan for sensitive information" in rendered
    forbidden = (
        "API Online",
        "API version",
        "Database Ready",
        "System Status",
        "Refresh backend status",
        "Built with FastAPI",
    )
    assert not any(term in rendered for term in forbidden)


def _render_privacy_four_detection_result_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    app.initialize_state()
    st.session_state.documents = [
        {
            "id": "doc-pii",
            "original_filename": "module12_4_pii.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "doc-manual",
            "original_filename": "manual_module5_employment.pdf",
            "status": "ready",
            "is_indexed": True,
        },
    ]
    st.session_state.last_pii_scan = {
        "document": {"document_id": "doc-pii", "filename": "module12_4_pii.pdf"},
        "detection_count": 4,
        "counts_by_type": {"email": 1, "phone_number": 1, "iban": 1, "credit_card": 1},
        "total_time_ms": 12.5,
        "detections": [
            {
                "detection_id": "pii-email",
                "pii_type": "email",
                "matched_text": "privacy.test@example.com",
                "page_number": 1,
                "redactable": True,
            },
            {
                "detection_id": "pii-phone",
                "pii_type": "phone_number",
                "matched_text": "+1 (202) 555-0147",
                "page_number": 1,
                "redactable": True,
            },
            {
                "detection_id": "pii-iban",
                "pii_type": "iban",
                "matched_text": "GB82 WEST 1234 5698 7654 32",
                "page_number": 1,
                "redactable": True,
            },
            {
                "detection_id": "pii-card",
                "pii_type": "credit_card",
                "matched_text": "4111 1111 1111 1111",
                "page_number": 1,
                "redactable": False,
            },
        ],
    }
    st.session_state.last_pii_redaction = {
        "document": {"document_id": "doc-pii", "filename": "module12_4_pii.pdf"},
        "redacted_count": 2,
        "artifact": {
            "artifact_id": "artifact-1",
            "filename": "redacted-module12_4_pii.pdf",
            "download_url": "/api/v1/documents/doc-pii/pii/artifacts/artifact-1",
        },
        "redaction_time_ms": 3.4,
        "total_time_ms": 4.2,
    }

    class PrivacyUiClient:
        def get_bytes(self, path: str, **_: object) -> bytes:
            assert path == "/api/v1/documents/doc-pii/pii/artifacts/artifact-1"
            return b"%PDF-redacted"

    app.render_privacy(PrivacyUiClient())


def test_privacy_page_renders_detection_review_and_artifact_download() -> None:
    app_test = AppTest.from_function(_render_privacy_four_detection_result_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    rendered = "\n".join(item.value for item in app_test.markdown)
    assert "Total detections" in rendered
    assert "privacy.test@example.com" in rendered
    assert "+1 (202) 555-0147" in rendered
    assert "GB82 WEST 1234 5698 7654 32" in rendered
    assert "4111 1111 1111 1111" in rendered
    assert "Safe to redact" in rendered
    assert "Review only" in rendered
    assert "Create redacted copy" in rendered
    assert "Redacted PDF created" in rendered
    assert any(item.label == "Select for redaction" for item in app_test.checkbox)
    assert any(item.label == "Not safely redactable" for item in app_test.checkbox)
    assert any(item.label == "Create redacted PDF" for item in app_test.button)
    assert len(app_test.download_button) == 1
    assert app_test.download_button[0].label == "Download redacted PDF"


def _render_privacy_no_pii_for_app_test() -> None:
    import streamlit as st

    from streamlit_app import app

    app.initialize_state()
    st.session_state.documents = [
        {
            "id": "doc-pii",
            "original_filename": "module12_4_pii.pdf",
            "status": "ready",
            "is_indexed": True,
        },
        {
            "id": "doc-manual",
            "original_filename": "manual_module5_employment.pdf",
            "status": "ready",
            "is_indexed": True,
        },
    ]
    st.session_state["privacy-document"] = "doc-manual"
    st.session_state.last_pii_scan = {
        "document": {"document_id": "doc-manual", "filename": "manual_module5_employment.pdf"},
        "detection_count": 0,
        "counts_by_type": {"email": 0, "phone_number": 0, "iban": 0, "credit_card": 0},
        "total_time_ms": 5.0,
        "detections": [],
    }
    class PrivacyUiClient:
        pass

    app.render_privacy(PrivacyUiClient())


def test_privacy_page_renders_controlled_no_pii_state() -> None:
    app_test = AppTest.from_function(_render_privacy_no_pii_for_app_test)
    app_test.run(timeout=10)

    assert not app_test.exception
    rendered = "\n".join(item.value for item in app_test.markdown)
    assert "No supported PII detected" in rendered
    assert "No high-confidence email, phone number, IBAN or credit-card values" in rendered
    assert not app_test.checkbox
