"""Deterministic tests for selected-document Streamlit scope helpers."""

from __future__ import annotations

from streamlit_app.app import (
    ALL_DOCUMENT_SCOPE,
    SELECTED_DOCUMENT_SCOPE,
    _document_scope_filter,
    _document_scope_labels,
)


FIRST_ID = "b5890e11-a025-4021-a222-c46a07b97ab5"
SECOND_ID = "62b74b1d-a6da-4520-b985-8623f966f2cb"


def make_document(document_id: str, filename: str, *, indexed: bool = True) -> dict[str, object]:
    return {
        "id": document_id,
        "original_filename": filename,
        "status": "ready",
        "is_indexed": indexed,
    }


def test_single_selected_filename_maps_to_exact_document_id() -> None:
    labels = _document_scope_labels(
        [make_document(FIRST_ID, "manual_module5_employment.pdf")]
    )

    assert labels[FIRST_ID].startswith("manual_module5_employment.pdf")
    assert _document_scope_filter(SELECTED_DOCUMENT_SCOPE, [FIRST_ID]) == {
        "document_ids": [FIRST_ID]
    }


def test_multiple_selection_preserves_order_and_removes_duplicate_ids() -> None:
    assert _document_scope_filter(
        SELECTED_DOCUMENT_SCOPE,
        [FIRST_ID, SECOND_ID, FIRST_ID],
    ) == {"document_ids": [FIRST_ID, SECOND_ID]}


def test_all_documents_ignores_stale_selection_and_unindexed_documents_are_hidden() -> None:
    labels = _document_scope_labels(
        [
            make_document(FIRST_ID, "manual_module5_employment.pdf"),
            make_document(SECOND_ID, "module9-evaluation.pdf", indexed=False),
        ]
    )

    assert list(labels) == [FIRST_ID]
    assert _document_scope_filter(ALL_DOCUMENT_SCOPE, [FIRST_ID, SECOND_ID]) is None


def test_duplicate_filenames_receive_distinct_display_labels() -> None:
    labels = _document_scope_labels(
        [
            make_document(FIRST_ID, "duplicate.pdf"),
            make_document(SECOND_ID, "duplicate.pdf"),
        ]
    )

    assert labels[FIRST_ID] != labels[SECOND_ID]
    assert FIRST_ID[:8] in labels[FIRST_ID]
    assert SECOND_ID[:8] in labels[SECOND_ID]
