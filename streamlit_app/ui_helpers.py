"""Pure helpers for predictable Streamlit state and user-facing errors."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from streamlit_app.api.client import ApiError


DOCUMENT_RESULT_KEYS = (
    "last_search",
    "last_rag",
    "last_summary",
    "last_classification",
    "last_extraction",
    "last_table_query",
    "last_comparison",
    "last_pii_scan",
    "last_pii_redaction",
)


def upload_size_error(filename: str, content_size: int, maximum_mb: int) -> str | None:
    """Return a friendly client-side error when a PDF exceeds the backend limit."""

    maximum_bytes = maximum_mb * 1024 * 1024
    if content_size <= maximum_bytes:
        return None
    return f"{filename} exceeds the {maximum_mb} MB per-document upload limit."


def conversation_display_labels(conversations: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Map conversation IDs to readable labels without exposing raw UUIDs."""

    titles: dict[str, int] = {}
    for item in conversations:
        title = str(item.get("title") or "Untitled conversation").strip()
        titles[title] = titles.get(title, 0) + 1

    labels: dict[str, str] = {}
    for index, item in enumerate(conversations, start=1):
        conversation_id = str(item.get("id", ""))
        title = str(item.get("title") or "Untitled conversation").strip()
        if titles[title] == 1:
            labels[conversation_id] = title
            continue
        timestamp = item.get("created_at") or item.get("updated_at")
        if timestamp:
            timestamp_text = str(timestamp).replace("T", " ").replace("Z", "")
            labels[conversation_id] = f"{title} · {timestamp_text[:16]}"
        else:
            labels[conversation_id] = f"{title} · conversation {index}"
    return labels


def page_display_data(page: dict[str, Any]) -> dict[str, Any]:
    """Prepare safe page presentation data without altering extracted OCR text."""

    extraction_method = str(page.get("extraction_method") or "unknown")
    is_ocr = bool(page.get("ocr_applied")) or extraction_method.casefold() == "ocr"
    return {
        "is_ocr": is_ocr,
        "extraction_method": "OCR" if is_ocr else extraction_method,
        "quality_note": "OCR quality may vary for scanned documents." if is_ocr else None,
        "expander_label": "View extracted OCR text" if is_ocr else None,
        "text": page.get("text") or "No extracted text.",
    }


def set_selected_document(
    state: MutableMapping[str, Any],
    selected_document_id: str | None,
) -> None:
    """Set the active document and clear results belonging to another document."""

    previous = state.get("selected_document_id")
    if previous != selected_document_id:
        for key in DOCUMENT_RESULT_KEYS:
            state[key] = None
    state["selected_document_id"] = selected_document_id


def friendly_api_error_message(error: ApiError) -> str:
    """Map transport/API failures to concise, safe presentation-layer copy."""

    if error.status_code == 404:
        return "The requested DocuIntel resource was not found. Refresh the page and try again."
    if error.status_code == 422:
        return f"Please check the submitted values: {error.message}"
    if error.status_code == 503 or "unavailable" in error.message.lower():
        return "The local DocuIntel service is unavailable. Start FastAPI and try again."
    if error.status_code is not None and error.status_code >= 500:
        return "The DocuIntel service could not complete that operation. Check the backend logs."
    return error.message


def prepare_source_rows(sources: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Prepare compact source metadata without inventing absent page information."""

    rows: list[dict[str, Any]] = []
    for source in sources or []:
        row: dict[str, Any] = {
            "Source": source.get("source_id") or source.get("label") or "Source",
            "Document": source.get("filename") or source.get("title") or "—",
            "Excerpt": source.get("excerpt") or "",
        }
        if source.get("page_number") is not None:
            row["Page"] = source["page_number"]
        elif source.get("start_page") is not None:
            row["Page"] = source["start_page"]
            if source.get("end_page") not in (None, source["start_page"]):
                row["Page"] = f"{source['start_page']}–{source['end_page']}"
        for source_key, display_key in (
            ("document_id", "Document ID"),
            ("chunk_id", "Chunk ID"),
            ("final_rank", "Final rank"),
            ("reranker_score", "Reranker score"),
        ):
            if source.get(source_key) is not None:
                row[display_key] = source[source_key]
        rows.append(row)
    return rows
