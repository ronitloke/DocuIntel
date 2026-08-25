"""DocuIntel Streamlit presentation layer.

This module deliberately communicates only with the public FastAPI HTTP API.
It does not import backend services, database models, vector stores, or Ollama.
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

from streamlit_app.api import analysis as analysis_api
from streamlit_app.api import comparison as comparison_api
from streamlit_app.api import conversations as conversations_api
from streamlit_app.api import documents as documents_api
from streamlit_app.api import privacy as privacy_api
from streamlit_app.api import rag as rag_api
from streamlit_app.api import search as search_api
from streamlit_app.api import tables as tables_api
from streamlit_app.api.client import ApiClient, ApiError
from streamlit_app.components.search_results import render_search_results
from streamlit_app.components.sources import render_sources
from streamlit_app.components.status import get_backend_status, render_backend_status, render_system_status
from streamlit_app.config import get_settings
from streamlit_app.evaluation import render_evaluation
from streamlit_app.ui_helpers import (
    conversation_display_labels,
    friendly_api_error_message,
    page_display_data,
    set_selected_document,
    upload_size_error,
)


PAGES = ("Home", "Documents", "Ask", "Analyze", "Compare", "Privacy", "Evaluation")
SEARCH_MODES = ("hybrid", "semantic", "keyword")
ALL_DOCUMENT_SCOPE = "All documents"
SELECTED_DOCUMENT_SCOPE = "Selected documents"
NO_STRUCTURED_TABLES_MESSAGE = "No structured tables were detected for this document."


@st.cache_resource(show_spinner=False)
def get_api_client() -> ApiClient:
    settings = get_settings()
    return ApiClient(settings.api_base_url, timeout=settings.api_timeout_seconds)


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "page": "Home",
        "documents": None,
        "conversations": None,
        "selected_document_id": None,
        "selected_conversation_id": None,
        "conversation_messages": [],
        "loaded_conversation_id": None,
        "last_search": None,
        "last_rag": None,
        "last_conversation_response": None,
        "last_summary": None,
        "last_classification": None,
        "last_extraction": None,
        "last_table_query": None,
        "last_comparison": None,
        "last_pii_scan": None,
        "last_pii_redaction": None,
        "backend_status": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("page") not in PAGES:
        st.session_state.page = "Home"


def main() -> None:
    st.set_page_config(page_title="DocuIntel", page_icon=":material/description:", layout="wide")
    initialize_state()
    client = get_api_client()

    st.sidebar.title("DocuIntel")
    st.sidebar.caption("Intelligent document processing & grounded RAG")
    page = st.sidebar.radio("Navigate", PAGES, key="page")
    if st.sidebar.button("Refresh backend status") or st.session_state.backend_status is None:
        st.session_state.backend_status = get_backend_status(client)
    render_backend_status(st.session_state.backend_status)
    st.sidebar.caption(f"API: {get_settings().api_base_url}")

    try:
        if page == "Home":
            render_home(client)
        elif page == "Documents":
            render_documents(client)
        elif page == "Ask":
            render_ask(client)
        elif page == "Analyze":
            render_analyze(client)
        elif page == "Compare":
            render_compare(client)
        elif page == "Privacy":
            render_privacy(client)
        elif page == "Evaluation":
            render_evaluation()
        else:
            render_home(client)
    except ApiError as exc:
        st.error(friendly_api_error_message(exc))


def render_home(client: ApiClient) -> None:
    st.title("DocuIntel")
    st.subheader("Intelligent document processing & grounded RAG platform")
    st.write(
        "Transform PDFs into searchable, explainable evidence. DocuIntel combines extraction, "
        "OCR, structure-aware indexing, hybrid retrieval, reranking, and source-aware answers "
        "behind a clear HTTP API."
    )
    capabilities = (
        ("Document intelligence", "Extract PDFs, OCR scanned pages, and inspect structure."),
        ("Grounded RAG", "Ask questions with retrieved evidence, reranking, and citations."),
        ("Structured analysis", "Extract fields and query persisted tables safely."),
        ("Document comparison", "Review added, removed, and modified evidence."),
        ("Privacy workflow", "Detect high-confidence PII and review redaction artifacts."),
        ("Benchmark evaluation", "Read the authoritative E5 quality and reliability package."),
    )
    for start in range(0, len(capabilities), 3):
        columns = st.columns(3)
        for column, (title, description) in zip(columns, capabilities[start : start + 3], strict=True):
            with column.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
    with st.container(border=True):
        st.subheader("Architecture at a glance")
        st.code(
            "PDF → extraction/OCR → structure-aware chunks → embeddings + PostgreSQL/pgvector\n"
            "→ hybrid retrieval → CrossEncoder → Ollama → grounded answer + citations",
            language="text",
        )
    render_system_status(st.session_state.backend_status or {}, ollama_model=get_settings().ollama_model)
    if st.button("Load document count"):
        response = documents_api.list_documents(client)
        st.session_state.documents = response.get("items", [])
        st.metric("Persisted documents", response.get("total", len(st.session_state.documents)))


def render_documents(client: ApiClient) -> None:
    st.title("Documents")
    st.caption("Upload, inspect, index, and remove documents through the FastAPI API.")
    if st.button("Refresh document list") or st.session_state.documents is None:
        st.session_state.documents = documents_api.list_documents(client).get("items", [])

    with st.form("upload_pdf_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "PDF files — maximum 25 MB per document",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDFs. Each file is validated against the existing 25 MB backend limit.",
        )
        upload_clicked = st.form_submit_button("Upload selected PDFs")
    if upload_clicked:
        if not uploaded_files:
            st.warning("Choose at least one PDF before uploading.")
        else:
            upload_results: list[tuple[str, bool, str]] = []
            with st.spinner("Uploading and extracting selected PDFs..."):
                for uploaded in uploaded_files:
                    content = uploaded.getvalue()
                    size_error = upload_size_error(
                        uploaded.name,
                        len(content),
                        get_settings().max_upload_size_mb,
                    )
                    if size_error:
                        upload_results.append((uploaded.name, False, size_error))
                        continue
                    try:
                        response = documents_api.upload_document(client, uploaded.name, content)
                    except ApiError as exc:
                        upload_results.append((uploaded.name, False, friendly_api_error_message(exc)))
                    else:
                        display_name = response.get("original_filename", uploaded.name)
                        upload_results.append((uploaded.name, True, f"{display_name} — uploaded"))
            for filename, succeeded, message in upload_results:
                if succeeded:
                    st.success(message)
                else:
                    st.error(f"{filename} — failed: {message}")
            st.session_state.documents = documents_api.list_documents(client).get("items", [])

    items = st.session_state.documents or []
    if not items:
        st.info("No documents are currently visible through the API.")
        return

    labels = {str(item["id"]): _document_label(item) for item in items}
    ids = list(labels)
    current = st.session_state.selected_document_id
    index = ids.index(current) if current in ids else 0
    selected_id = st.selectbox("Select a document", ids, index=index, format_func=labels.get)
    set_selected_document(st.session_state, selected_id)
    render_document_detail(client, selected_id)


def render_document_detail(client: ApiClient, document_id: str) -> None:
    detail = documents_api.get_document(client, document_id)
    st.divider()
    st.subheader(detail.get("original_filename", "Document detail"))
    cols = st.columns(4)
    cols[0].metric("Status", detail.get("status", "unknown"))
    cols[1].metric("Pages", detail.get("page_count", "—"))
    cols[2].metric("Chunks", detail.get("chunk_count", "—"))
    cols[3].metric("Indexed", "Yes" if detail.get("is_indexed") else "No")
    metadata = detail.get("metadata") or {}
    summary = detail.get("summary") or {}
    with st.expander("Metadata and processing summary"):
        st.json({"metadata": metadata, "summary": summary})

    action_cols = st.columns(2)
    if action_cols[0].button("Index / rebuild chunks", key=f"index-{document_id}"):
        with st.spinner("Chunking and embedding document..."):
            response = documents_api.index_document(client, document_id)
        st.success(f"Indexed {response.get('chunks_created', 0)} chunks.")
        st.session_state.documents = documents_api.list_documents(client).get("items", [])
    with st.form(f"delete-document-{document_id}"):
        confirmed = st.checkbox(
            "I understand this deletes the document",
            key=f"confirm-delete-{document_id}",
        )
        delete_clicked = st.form_submit_button("Delete document")
    if delete_clicked:
        if not confirmed:
            st.warning("Confirm deletion first.")
        else:
            documents_api.delete_document(client, document_id)
            st.success("Document deleted.")
            st.session_state.selected_document_id = None
            st.session_state.documents = documents_api.list_documents(client).get("items", [])
            st.rerun()

    pages = documents_api.list_pages(client, document_id).get("items", [])
    chunks = documents_api.list_chunks(client, document_id).get("items", [])
    page_tab, chunk_tab = st.tabs([f"Pages ({len(pages)})", f"Chunks ({len(chunks)})"])
    with page_tab:
        render_pages(client, document_id, pages)
    with chunk_tab:
        if not chunks:
            st.info("No chunks are available. Index the document to create searchable chunks.")
        for chunk in chunks:
            with st.expander(
                f"Chunk {chunk.get('sequence_number', '—')} · "
                f"pages {chunk.get('start_page', '—')}–{chunk.get('end_page', '—')}"
            ):
                st.write(chunk.get("text") or "")
                st.caption(
                    f"{chunk.get('content_type', 'unknown')} · "
                    f"{chunk.get('character_count', '—')} characters"
                )


def render_analyze(client: ApiClient) -> None:
    """Render transient summary and classification controls for one document."""

    st.title("Analyze")
    st.caption(
        "Generate grounded summaries and caller-constrained classifications from indexed document chunks."
    )
    documents = _load_documents_for_filters(client)
    if not documents:
        st.info("No documents are currently visible through the API.")
        return

    labels = {str(item["id"]): _document_label(item) for item in documents}
    document_ids = list(labels)
    current = st.session_state.selected_document_id
    index = document_ids.index(current) if current in document_ids else 0
    selected_id = st.selectbox(
        "Select a document",
        document_ids,
        index=index,
        format_func=labels.get,
        key="analyze-document",
    )
    set_selected_document(st.session_state, selected_id)
    detail = documents_api.get_document(client, selected_id)
    info_cols = st.columns(4)
    info_cols[0].metric("Filename", detail.get("original_filename", "—"))
    info_cols[1].metric("Pages", detail.get("page_count", "—"))
    info_cols[2].metric("Chunks", detail.get("chunk_count", "—"))
    info_cols[3].metric("Indexed", "Yes" if detail.get("is_indexed") else "No")

    indexed_documents = [
        item
        for item in documents
        if item.get("status") == "ready" and item.get("is_indexed") is True
    ]
    summary_tab, classification_tab, extraction_tab, tables_tab = st.tabs(
        ["Summary", "Classification", "Extraction", "Tables"],
        key="analyze-tabs",
    )
    with summary_tab:
        with st.container(border=True):
            st.subheader("Summarize document")
            with st.form("document-summary-form"):
                style = st.selectbox(
                    "Summary style",
                    ["brief", "detailed", "bullet_points"],
                    format_func=lambda value: value.replace("_", " ").title(),
                )
                summary_clicked = st.form_submit_button("Generate summary")
            if summary_clicked:
                with st.spinner("Generating grounded document summary..."):
                    st.session_state.last_summary = analysis_api.summarize(
                        client,
                        selected_id,
                        style=style,
                    )
            summary = st.session_state.last_summary
            if summary and str(summary.get("document_id")) == selected_id:
                st.markdown(summary.get("summary") or "No summary was returned.")
                render_analysis_sources(summary.get("sources"))
                render_timings(
                    summary,
                    (
                        "content_loading_time_ms",
                        "partial_generation_time_ms",
                        "final_synthesis_time_ms",
                        "generation_time_ms",
                        "grounding_verification_time_ms",
                        "grounding_repair_time_ms",
                        "grounding_verification_passes",
                        "total_time_ms",
                    ),
                )

    with classification_tab:
        with st.container(border=True):
            st.subheader("Classify document")
            st.caption("Enter at least two allowed labels, one per line. Labels are not persisted.")
            with st.form("document-classification-form"):
                label_text = st.text_area(
                    "Allowed labels",
                    value="Employment Policy\nExpense Policy\nOther",
                    height=120,
                )
                classify_clicked = st.form_submit_button("Classify document")
            if classify_clicked:
                labels_input = [line.strip() for line in label_text.splitlines() if line.strip()]
                if len(labels_input) < 2:
                    st.warning("Enter at least two distinct labels.")
                else:
                    with st.spinner("Classifying document from grounded evidence..."):
                        st.session_state.last_classification = analysis_api.classify(
                            client,
                            selected_id,
                            labels=labels_input,
                        )
            classification = st.session_state.last_classification
            if classification and str(classification.get("document_id")) == selected_id:
                st.success(f"Selected label: {classification.get('selected_label', '—')}")
                st.write(classification.get("rationale") or "No rationale was returned.")
                render_analysis_sources(classification.get("sources"))
                render_timings(
                    classification,
                    ("content_loading_time_ms", "generation_time_ms", "total_time_ms"),
                )

    with extraction_tab:
        render_structured_extraction(client, indexed_documents, selected_id)
    with tables_tab:
        render_table_query(client, indexed_documents, selected_id)


def render_compare(client: ApiClient) -> None:
    """Render the minimal Module 12.3 base/target comparison workflow."""

    st.title("Compare")
    st.caption(
        "Compare exactly two ready indexed documents. Version mode treats Base as the older document and Target as the newer document."
    )
    documents = [
        item
        for item in _load_documents_for_filters(client)
        if item.get("status") == "ready" and item.get("is_indexed") is True
    ]
    if len(documents) < 2:
        st.info("At least two ready indexed documents are required for comparison.")
        return

    labels = {str(item["id"]): _document_label(item) for item in documents}
    document_ids = list(labels)
    with st.container(border=True):
        with st.form("comparison-form"):
            base_id = st.selectbox(
                "Base document",
                document_ids,
                format_func=labels.get,
                key="comparison-base-document",
            )
            target_id = st.selectbox(
                "Target document",
                document_ids,
                format_func=labels.get,
                key="comparison-target-document",
            )
            mode = st.selectbox(
                "Comparison mode",
                ["document", "version"],
                format_func=lambda value: "Version comparison" if value == "version" else "Document comparison",
                key="comparison-mode",
            )
            include_tables = st.checkbox("Compare structured tables", value=True, key="comparison-tables")
            include_unchanged = st.checkbox("Include unchanged items", value=False, key="comparison-unchanged")
            generate_summary = st.checkbox("Generate grounded change summary", value=True, key="comparison-summary")
            compare_clicked = st.form_submit_button("Compare documents", type="primary")

    if compare_clicked:
        st.session_state.last_comparison = None
        try:
            with st.spinner("Comparing stored document evidence..."):
                st.session_state.last_comparison = comparison_api.compare_documents(
                    client,
                    base_document_id=base_id,
                    target_document_id=target_id,
                    mode=mode,
                    include_tables=include_tables,
                    include_unchanged=include_unchanged,
                    generate_summary=generate_summary,
                )
        except ApiError as exc:
            st.error(format_module_12_3_api_error(exc))

    response = st.session_state.last_comparison
    if not response:
        return
    if (
        str(response.get("base_document", {}).get("document_id")) != base_id
        or str(response.get("target_document", {}).get("document_id")) != target_id
    ):
        return

    base_filename = response.get("base_document", {}).get("filename", "Base document")
    target_filename = response.get("target_document", {}).get("filename", "Target document")
    st.subheader(f"{base_filename} → {target_filename}")
    statistics = response.get("statistics", {})
    metric_columns = st.columns(5)
    metric_columns[0].metric("Added", statistics.get("added_count", 0))
    metric_columns[1].metric("Removed", statistics.get("removed_count", 0))
    metric_columns[2].metric("Modified", statistics.get("modified_count", 0))
    metric_columns[3].metric("Unchanged", statistics.get("unchanged_count", 0))
    metric_columns[4].metric("Table changes", statistics.get("table_change_count", 0))
    st.markdown(response.get("summary") or "No change summary was returned.")

    changes = response.get("changes", [])
    for change_type, title in (("modified", "Modified"), ("added", "Added"), ("removed", "Removed"), ("unchanged", "Unchanged")):
        matching = [change for change in changes if change.get("change_type") == change_type]
        if not matching:
            continue
        with st.expander(f"{title} ({len(matching)})", expanded=change_type != "unchanged"):
            for change in matching:
                st.markdown(
                    f"**{change.get('change_id', 'Change')}** · {change.get('scope', 'text')}"
                )
                before, after = st.columns(2)
                before.caption("Base")
                before.write(change.get("base_text") or "—")
                after.caption("Target")
                after.write(change.get("target_text") or "—")
                if change.get("table_detail"):
                    st.json(change["table_detail"])
                provenance = [
                    *(change.get("base_provenance") or []),
                    *(change.get("target_provenance") or []),
                ]
                if provenance:
                    st.caption(
                        " · ".join(
                            f"{item.get('source_id')} · {item.get('filename')} · page {item.get('page_number', '—')}"
                            for item in provenance
                        )
                    )
    render_timings(
        response,
        (
            "content_loading_time_ms",
            "alignment_time_ms",
            "table_comparison_time_ms",
            "summary_generation_time_ms",
            "total_time_ms",
        ),
    )


def render_privacy(client: ApiClient) -> None:
    """Render the review-first, explicit-selection privacy workflow."""

    st.title("Privacy")
    st.caption(
        "Detect high-confidence structured PII locally, review each result, and create a new redacted PDF."
    )
    documents = [
        item
        for item in _load_documents_for_filters(client)
        if item.get("status") == "ready"
    ]
    if not documents:
        st.info("No ready documents are available for privacy scanning.")
        return

    labels = {str(item["id"]): _document_label(item) for item in documents}
    document_ids = list(labels)
    with st.container(border=True):
        with st.form("privacy-scan-form"):
            document_id = st.selectbox(
                "Document",
                document_ids,
                format_func=labels.get,
                key="privacy-document",
            )
            pii_types = st.multiselect(
                "PII types",
                ["email", "phone_number", "iban", "credit_card"],
                default=["email", "phone_number", "iban", "credit_card"],
                key="privacy-types",
            )
            scan_clicked = st.form_submit_button("Scan for PII", type="primary")
    if scan_clicked:
        st.session_state.last_pii_scan = None
        st.session_state.last_pii_redaction = None
        if not pii_types:
            st.warning("Choose at least one PII type before scanning.")
        else:
            try:
                with st.spinner("Scanning document locally for validated PII..."):
                    st.session_state.last_pii_scan = privacy_api.detect_pii(
                        client,
                        document_id,
                        pii_types,
                    )
            except ApiError as exc:
                st.error(format_module_12_4_api_error(exc, action="scan"))

    scan = st.session_state.last_pii_scan
    if not scan or str(scan.get("document", {}).get("document_id")) != document_id:
        return

    counts = scan.get("counts_by_type", {})
    count_columns = st.columns(4)
    for column, pii_type in zip(
        count_columns,
        ("email", "phone_number", "iban", "credit_card"),
        strict=False,
    ):
        column.metric(pii_type.replace("_", " ").title(), counts.get(pii_type, 0))
    st.caption(
        f"{scan.get('detection_count', 0)} high-confidence detections · "
        f"scan time {scan.get('total_time_ms', 0):.1f} ms"
    )
    detections = scan.get("detections", [])
    if not detections:
        st.info("No high-confidence supported PII was detected in this document.")
        return

    redactable_count = sum(bool(detection.get("redactable")) for detection in detections)
    if not redactable_count:
        st.warning("Detections were found, but none have verified page coordinates for safe redaction.")

    with st.container(border=True):
        st.subheader("Review detections")
        st.caption("Select only the values you explicitly want removed. Scanning never redacts automatically.")
        with st.form("privacy-redaction-form"):
            selected_ids: list[str] = []
            for detection in detections:
                detection_id = str(detection.get("detection_id", ""))
                redactable = bool(detection.get("redactable"))
                label = (
                    f"{detection.get('pii_type', 'pii')} · page {detection.get('page_number', '—')} · "
                    f"{detection.get('matched_text', '')}"
                )
                if st.checkbox(
                    label,
                    key=f"privacy-select-{detection_id}",
                    disabled=not redactable,
                ):
                    selected_ids.append(detection_id)
                st.caption(
                    f"{detection_id} · {'redactable' if redactable else 'not redactable: exact coordinates unavailable'}"
                )
            redact_clicked = st.form_submit_button("Create redacted PDF", type="primary")
        if redact_clicked:
            if not selected_ids:
                st.warning("Select at least one redactable detection first.")
            else:
                try:
                    with st.spinner("Applying and verifying irreversible PDF redactions..."):
                        st.session_state.last_pii_redaction = privacy_api.redact_pii(
                            client,
                            document_id,
                            selected_ids,
                        )
                except ApiError as exc:
                    st.error(format_module_12_4_api_error(exc, action="redaction"))

    redaction = st.session_state.last_pii_redaction
    if redaction and str(redaction.get("document", {}).get("document_id")) == document_id:
        artifact = redaction.get("artifact") or {}
        st.success(
            f"Created a new redacted PDF with {redaction.get('redacted_count', 0)} selected detection(s). "
            "The original PDF was not modified."
        )
        try:
            artifact_bytes = privacy_api.download_redacted_artifact(
                client,
                str(artifact.get("download_url", "")),
            )
            st.download_button(
                "Download redacted PDF",
                data=artifact_bytes,
                file_name=artifact.get("filename", "redacted-document.pdf"),
                mime="application/pdf",
                key=f"privacy-download-{artifact.get('artifact_id', 'artifact')}",
            )
        except ApiError as exc:
            st.error(format_module_12_4_api_error(exc, action="download"))
        render_timings(
            redaction,
            ("redaction_time_ms", "total_time_ms"),
        )


def format_module_12_4_api_error(exc: ApiError, *, action: str) -> str:
    """Map privacy failures to concise, non-traceback UI messages."""

    action_labels = {"scan": "PII scan", "redaction": "PDF redaction", "download": "Download"}
    label = action_labels.get(action, "Privacy operation")
    if exc.status_code == 404:
        return f"{label} could not find the selected document or generated artifact."
    if exc.status_code == 422:
        return f"{label} was rejected: {exc.message}"
    if exc.status_code is not None and exc.status_code >= 500:
        return f"{label} failed safely. Check that the backend and database are ready."
    return exc.message


def format_module_12_3_api_error(exc: ApiError) -> str:
    """Map comparison transport failures to concise user-facing messages."""

    if exc.status_code == 404:
        return "One of the selected documents was not found. Refresh the document list and try again."
    if exc.status_code == 422:
        return f"Comparison request was rejected: {exc.message}"
    if exc.status_code is not None and exc.status_code >= 500:
        return "Comparison could not be completed. Check that the backend service is ready."
    return exc.message


SUPPORTED_EXTRACTION_TYPES = ("string", "integer", "number", "boolean", "date", "list[string]")


def parse_extraction_field_lines(text: str) -> tuple[list[dict[str, str | None]], list[str]]:
    """Parse the small line-oriented extraction UX without evaluating user text."""

    fields: list[dict[str, str | None]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) > 3 or not parts[0]:
            errors.append(f"Line {line_number}: use name | type | optional description.")
            continue
        name = parts[0]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            errors.append(f"Line {line_number}: field names must use letters, digits, and underscores.")
            continue
        field_type = parts[1] if len(parts) >= 2 and parts[1] else "string"
        if field_type not in SUPPORTED_EXTRACTION_TYPES:
            errors.append(f"Line {line_number}: unsupported field type '{field_type}'.")
            continue
        key = name.casefold()
        if key in seen:
            errors.append(f"Line {line_number}: duplicate field name '{name}'.")
            continue
        seen.add(key)
        fields.append(
            {
                "name": name,
                "type": field_type,
                "description": parts[2] if len(parts) == 3 and parts[2] else None,
            }
        )
    if not fields and not errors:
        errors.append("Enter at least one requested field.")
    return fields, errors


def render_structured_extraction(
    client: ApiClient,
    indexed_documents: list[dict[str, Any]],
    selected_id: str,
) -> None:
    """Render minimal caller-defined extraction controls and provenance."""

    with st.container(border=True):
        st.subheader("Structured extraction")
        st.caption("Extract bounded typed fields from the selected indexed document.")
        if not indexed_documents or selected_id not in {str(item["id"]) for item in indexed_documents}:
            st.info("Select an indexed document to use structured extraction.")
            return
        with st.form("structured-extraction-form"):
            field_text = st.text_area(
                "Requested fields",
                value=(
                    "notice_period | string | Required resignation notice period\n"
                    "invoice_reference | string | Invoice/reference identifier\n"
                    "employee_name | string | Employee name"
                ),
                height=140,
                key="structured-extraction-fields",
            )
            extraction_clicked = st.form_submit_button("Run extraction")
        if extraction_clicked:
            fields, errors = parse_extraction_field_lines(field_text)
            if errors:
                for error in errors:
                    st.warning(error)
            else:
                st.session_state.last_extraction = None
                try:
                    with st.spinner("Extracting structured evidence..."):
                        st.session_state.last_extraction = analysis_api.extract(
                            client,
                            selected_id,
                            fields=fields,
                        )
                except ApiError as exc:
                    st.error(format_module_12_2_api_error(exc, action="extraction"))
        response = st.session_state.last_extraction
        if response and str(response.get("document_id")) == selected_id:
            for field in response.get("fields", []):
                status = field.get("status", "not_found")
                value = field.get("value")
                with st.expander(f"{field.get('field', 'field')} · {status}"):
                    st.write(value if value is not None else "Not found in the supplied evidence.")
                    if field.get("candidates"):
                        st.json({"candidates": field["candidates"]})
                    st.caption("Sources: " + (", ".join(field.get("sources", [])) or "none"))
            render_analysis_sources(response.get("sources"))
            render_timings(
                response,
                (
                    "evidence_loading_time_ms",
                    "generation_time_ms",
                    "validation_time_ms",
                    "total_time_ms",
                ),
            )


def render_table_query(
    client: ApiClient,
    indexed_documents: list[dict[str, Any]],
    selected_id: str,
) -> None:
    """Render table inventory, bounded preview, and safe natural-language query."""

    with st.container(border=True):
        st.subheader("Table query")
        st.caption("Inspect extracted tables and run deterministic, source-mapped operations.")
        if not indexed_documents or selected_id not in {str(item["id"]) for item in indexed_documents}:
            st.info("Select an indexed document to use table querying.")
            return
        try:
            table_inventory = tables_api.list_tables(client, selected_id).get("tables", [])
        except ApiError as exc:
            st.error(format_module_12_2_api_error(exc, action="table_inventory"))
            return
        empty_message = structured_table_empty_state(table_inventory)
        if empty_message:
            st.info(empty_message)
            return
        table_labels = {
            str(table["table_id"]): (
                f"Table {table.get('table_index', '—')} · page {table.get('page_number', '—')} · "
                f"{table.get('row_count', 0)} rows"
            )
            for table in table_inventory
        }
        table_ids = list(table_labels)
        table_id = st.selectbox(
            "Select a table",
            table_ids,
            format_func=table_labels.get,
            key=f"analyze-table-{selected_id}",
        )
        try:
            preview = tables_api.preview_table(client, selected_id, table_id, preview_rows=20)
        except ApiError as exc:
            st.error(format_module_12_2_api_error(exc, action="table_preview"))
            return
        headers = preview.get("headers", [])
        rows = preview.get("rows", [])
        if headers:
            st.table([dict(zip(headers, row, strict=False)) for row in rows])
        if preview.get("truncated"):
            st.caption("Preview truncated; supported operations use the bounded backend table data.")
        with st.form(f"table-query-form-{selected_id}"):
            st.caption(
                "Examples: Which product has the highest quantity? · "
                "What is the total price? · How many rows are there?"
            )
            question = st.text_input(
                "Table question",
                placeholder="Which product has the highest quantity?",
                key=f"table-query-question-{selected_id}",
            )
            query_clicked = st.form_submit_button("Run table query")
        if query_clicked:
            if not question.strip():
                st.warning("Enter a table question.")
            else:
                st.session_state.last_table_query = None
                try:
                    with st.spinner("Planning and executing table query..."):
                        st.session_state.last_table_query = tables_api.query_table(
                            client,
                            selected_id,
                            table_id,
                            question=question.strip(),
                        )
                except ApiError as exc:
                    st.error(format_module_12_2_api_error(exc, action="table_query"))
        response = st.session_state.last_table_query
        if response and str(response.get("document_id")) == selected_id and response.get("table", {}).get("table_id") == table_id:
            st.markdown(response.get("answer") or "No table answer was returned.")
            st.json(response.get("result", {}))
            for source in response.get("sources", []):
                st.caption(
                    f"{source.get('source_id', 'T1')} · {source.get('filename', 'document')} · "
                    f"page {source.get('page_number', '—')} · table {source.get('table_index', '—')} · "
                    f"rows {source.get('row_indices', [])}"
                )
            render_timings(
                response,
                (
                    "table_loading_time_ms",
                    "plan_generation_time_ms",
                    "execution_time_ms",
                    "total_time_ms",
                ),
            )


def format_module_12_2_api_error(exc: ApiError, *, action: str) -> str:
    """Map Module 12.2 transport/API failures to concise UI messages."""

    action_labels = {
        "extraction": "Structured extraction",
        "table_inventory": "Table inventory",
        "table_preview": "Table preview",
        "table_query": "Table query",
    }
    label = action_labels.get(action, "Request")
    if exc.status_code == 404:
        if action == "table_inventory":
            return (
                "The table endpoint was not found on the connected backend. "
                "Restart FastAPI to load the Module 12.2 routes."
            )
        return "Document or table not found. Refresh the document list and try again."
    if exc.status_code == 422:
        return f"{label} request was rejected: {exc.message}"
    if exc.status_code is not None and exc.status_code >= 500:
        return f"{label} could not be completed. Check that the backend service is ready."
    return exc.message


def structured_table_empty_state(table_inventory: list[dict[str, Any]]) -> str | None:
    """Return the normal empty-state message when a document has no tables."""

    return NO_STRUCTURED_TABLES_MESSAGE if not table_inventory else None


def render_analysis_sources(sources: Any) -> None:
    """Display compact provenance without exposing full document content."""

    if not sources:
        st.info("No evidence metadata was returned.")
        return
    with st.expander("Evidence and source metadata"):
        for source in sources:
            page = source.get("start_page", "—")
            if source.get("end_page") and source.get("end_page") != page:
                page = f"{page}–{source['end_page']}"
            st.markdown(
                f"**{source.get('source_id', 'Source')}** · "
                f"{source.get('filename', 'document')} · page {page}"
            )
            st.caption(
                f"Chunk sequence {source.get('sequence_number', '—')} · "
                f"{source.get('section_heading') or 'No section heading'}"
            )
            st.write(source.get("excerpt") or "")


def render_pages(client: ApiClient, document_id: str, pages: list[dict[str, Any]]) -> None:
    if not pages:
        st.info("No persisted page metadata is available.")
        return
    page_numbers = [int(page["page_number"]) for page in pages]
    page_number = st.selectbox("Page", page_numbers, key=f"page-select-{document_id}")
    page = documents_api.get_page(client, document_id, page_number)
    display = page_display_data(page)
    st.caption(f"Extraction method: {display['extraction_method']}")
    if display["is_ocr"]:
        st.caption(str(display["quality_note"]))
        with st.expander(str(display["expander_label"])):
            st.text(display["text"])
    else:
        st.write(display["text"])
    if page.get("layout_elements") or page.get("tables"):
        with st.expander("Structured page data"):
            st.json({"layout_elements": page.get("layout_elements", []), "tables": page.get("tables", [])})


def render_search(client: ApiClient, *, nested: bool = False) -> None:
    (st.subheader if nested else st.title)("Search evidence")
    st.caption("Use Module 5 retrieval and optionally Module 6 reranking.")
    documents = _load_documents_for_filters(client)
    scope_mode = render_document_scope_selector(key_prefix="search")
    with st.form("search_form"):
        query = st.text_input("Search query")
        mode = st.selectbox("Search mode", SEARCH_MODES, index=0)
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
        rerank = st.checkbox("Use CrossEncoder reranking", value=False)
        filters = render_filter_controls(
            documents,
            key_prefix="search",
            scope_mode=scope_mode,
        )
        submitted = st.form_submit_button("Search")
    if submitted:
        if filters and filters.get("document_ids") == []:
            st.warning("Select at least one document or choose All documents.")
        elif not query.strip():
            st.warning("Enter a search query.")
        else:
            with st.spinner("Searching indexed evidence..."):
                st.session_state.last_search = search_api.search(
                    client,
                    query=query.strip(),
                    mode=mode,
                    top_k=int(top_k),
                    rerank=rerank,
                    filters=filters,
                )
    response = st.session_state.last_search
    render_search_results(response)
    if response:
        render_timings(response, ("search_time_ms", "retrieval_time_ms", "rerank_time_ms", "total_search_time_ms"))


def render_ask(client: ApiClient) -> None:
    st.title("Ask")
    st.caption("Search evidence, ask grounded questions, or continue a persistent conversation.")
    answer_tab, search_tab, conversation_tab = st.tabs(
        ["Grounded Q&A", "Search evidence", "Conversations"]
    )
    with answer_tab:
        render_grounded_ask(client)
    with search_tab:
        render_search(client, nested=True)
    with conversation_tab:
        render_conversations(client, nested=True)


def render_grounded_ask(client: ApiClient) -> None:
    st.subheader("Grounded Q&A")
    st.caption("Answers are generated only from evidence returned by the backend retrieval pipeline.")
    documents = _load_documents_for_filters(client)
    scope_mode = render_document_scope_selector(key_prefix="ask")
    with st.form("ask_form"):
        question = st.text_area("Question", placeholder="What does the indexed policy say?")
        mode = st.selectbox("Search mode", SEARCH_MODES, index=0, key="ask-mode")
        top_k = st.number_input("Evidence top K", min_value=1, max_value=50, value=5, key="ask-top-k")
        rerank = st.checkbox("Use CrossEncoder reranking", value=True, key="ask-rerank")
        filters = render_filter_controls(
            documents,
            key_prefix="ask",
            scope_mode=scope_mode,
        )
        submitted = st.form_submit_button("Ask")
    if submitted:
        if filters and filters.get("document_ids") == []:
            st.warning("Select at least one document or choose All documents.")
        elif not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                st.session_state.last_rag = rag_api.ask(
                    client,
                    question=question.strip(),
                    top_k=int(top_k),
                    search_mode=mode,
                    rerank=rerank,
                    filters=filters,
                )
    response = st.session_state.last_rag
    if response:
        st.subheader("Answer")
        st.markdown(response.get("answer") or "No answer was returned.")
        if response.get("citations_valid") is False:
            st.warning("The backend flagged one or more unsupported source citations.")
        render_sources(response.get("sources"))
        render_timings(response, ("retrieval_time_ms", "rerank_time_ms", "generation_time_ms", "total_time_ms"))


def render_conversations(client: ApiClient, *, nested: bool = False) -> None:
    (st.subheader if nested else st.title)("Conversations")
    st.caption("Persistent multi-turn RAG sessions backed by the existing conversation API.")
    if st.button("Refresh conversations") or st.session_state.conversations is None:
        st.session_state.conversations = conversations_api.list_conversations(client)
    with st.form("create-conversation-form", clear_on_submit=True):
        title = st.text_input("Optional conversation title")
        create_clicked = st.form_submit_button("New conversation")
    if create_clicked:
        conversation = conversations_api.create_conversation(client, title.strip() or None)
        st.session_state.conversations = conversations_api.list_conversations(client)
        st.session_state.selected_conversation_id = str(conversation["id"])
        st.session_state.loaded_conversation_id = None
        st.session_state.conversation_messages = []
        st.rerun()

    conversations = st.session_state.conversations or []
    if not conversations:
        st.info("No conversations yet. Create one to start a persistent RAG session.")
        return
    labels = conversation_display_labels(conversations)
    ids = list(labels)
    current = st.session_state.selected_conversation_id
    index = ids.index(current) if current in ids else 0
    selected = st.selectbox("Conversation", ids, index=index, format_func=labels.get)
    st.session_state.selected_conversation_id = selected
    if st.session_state.loaded_conversation_id != selected:
        st.session_state.conversation_messages = conversations_api.list_messages(client, selected)
        st.session_state.loaded_conversation_id = selected
        st.session_state.last_conversation_response = None

    with st.form("delete-conversation-form"):
        confirmed = st.checkbox("I understand this deletes the conversation")
        delete_clicked = st.form_submit_button("Delete conversation")
    if delete_clicked:
        if not confirmed:
            st.warning("Confirm deletion first.")
        else:
            conversations_api.delete_conversation(client, selected)
            st.session_state.conversations = conversations_api.list_conversations(client)
            st.session_state.selected_conversation_id = None
            st.session_state.loaded_conversation_id = None
            st.session_state.conversation_messages = []
            st.rerun()

    for message in st.session_state.conversation_messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content") or "")
    documents = _load_documents_for_filters(client)
    conversation_filters = render_filter_controls(
        documents,
        key_prefix=f"conversation-{selected}",
    )
    question = st.chat_input("Ask a follow-up question")
    if question and question.strip():
        if conversation_filters and conversation_filters.get("document_ids") == []:
            st.warning("Select at least one document or choose All documents.")
        else:
            with st.spinner("Retrieving evidence and generating the next turn..."):
                response = conversations_api.ask_in_conversation(
                    client,
                    selected,
                    question=question.strip(),
                    top_k=5,
                    search_mode="hybrid",
                    rerank=True,
                    filters=conversation_filters,
                )
            st.session_state.last_conversation_response = response
            st.session_state.conversation_messages = conversations_api.list_messages(client, selected)
            st.rerun()
    response = st.session_state.last_conversation_response
    if response:
        render_sources(response.get("sources"))
        render_timings(
            response,
            ("history_load_time_ms", "query_rewrite_time_ms", "retrieval_time_ms", "generation_time_ms", "total_time_ms"),
        )


def render_document_scope_selector(*, key_prefix: str) -> str:
    """Render scope outside forms so changing it triggers an immediate rerun."""

    return st.selectbox(
        "Document scope",
        [ALL_DOCUMENT_SCOPE, SELECTED_DOCUMENT_SCOPE],
        key=f"{key_prefix}-document-scope",
    )


def _document_scope_labels(documents: list[dict[str, Any]]) -> dict[str, str]:
    """Return indexed document IDs mapped to stable, user-friendly labels."""

    indexed = [
        item
        for item in documents
        if item.get("status") == "ready" and item.get("is_indexed") is True
    ]
    base_labels = [(_document_label(item), str(item["id"])) for item in indexed]
    counts: dict[str, int] = {}
    for label, _ in base_labels:
        counts[label] = counts.get(label, 0) + 1

    labels: dict[str, str] = {}
    for label, document_id in base_labels:
        if counts[label] > 1:
            label = f"{label} · {document_id[:8]}"
        labels[document_id] = label
    return labels


def _document_scope_filter(scope_mode: str, selected_ids: list[str]) -> dict[str, Any] | None:
    """Build the typed-ID portion of a request without stale all-document selections."""

    if scope_mode != SELECTED_DOCUMENT_SCOPE:
        return None
    return {"document_ids": list(dict.fromkeys(selected_ids))}


def render_filter_controls(
    documents: list[dict[str, Any]],
    *,
    key_prefix: str,
    scope_mode: str | None = None,
) -> dict[str, Any] | None:
    labels = _document_scope_labels(documents)
    scope_mode = scope_mode or render_document_scope_selector(key_prefix=key_prefix)
    selection_key = f"{key_prefix}-document-selection"
    existing_selection = st.session_state.get(selection_key)
    if isinstance(existing_selection, list):
        st.session_state[selection_key] = [item for item in existing_selection if item in labels]
    selected_ids = st.multiselect(
        "Documents in scope",
        list(labels),
        format_func=lambda value: labels[value],
        disabled=scope_mode != SELECTED_DOCUMENT_SCOPE,
        key=selection_key,
    )
    content_types = st.multiselect(
        "Content types",
        ["text", "table", "list", "mixed"],
        key=f"{key_prefix}-content-types",
    )
    ocr_choice = st.selectbox("OCR chunks", ["Any", "Only OCR", "Exclude OCR"], key=f"{key_prefix}-ocr")
    page_cols = st.columns(2)
    page_start = page_cols[0].number_input("Page from", min_value=1, value=1, key=f"{key_prefix}-page-start")
    page_end = page_cols[1].number_input("Page to (0 = any)", min_value=0, value=0, key=f"{key_prefix}-page-end")
    filters: dict[str, Any] = {}
    scope_filter = _document_scope_filter(scope_mode, selected_ids)
    if scope_filter is not None:
        filters.update(scope_filter)
    if content_types:
        filters["content_types"] = content_types
    if ocr_choice != "Any":
        filters["contains_ocr"] = ocr_choice == "Only OCR"
    if page_start > 1:
        filters["page_start"] = int(page_start)
    if page_end > 0:
        filters["page_end"] = int(page_end)
    if scope_mode == ALL_DOCUMENT_SCOPE and filters:
        filters["document_ids"] = None
    return filters or None


def _load_documents_for_filters(client: ApiClient) -> list[dict[str, Any]]:
    if st.session_state.documents is None:
        st.session_state.documents = documents_api.list_documents(client).get("items", [])
    return st.session_state.documents or []


def render_timings(response: dict[str, Any], fields: tuple[str, ...]) -> None:
    values = {field: response.get(field) for field in fields if response.get(field) is not None}
    if values:
        with st.expander("Timing metadata"):
            st.json(values)


def _document_label(item: dict[str, Any]) -> str:
    return f"{item.get('original_filename', 'Unnamed document')} · {item.get('status', 'unknown')}"


def _conversation_label(item: dict[str, Any]) -> str:
    """Return a readable conversation title without exposing its UUID."""

    return conversation_display_labels([item]).get(
        str(item.get("id", "")),
        "Untitled conversation",
    )


if __name__ == "__main__":
    main()
