"""DocuIntel Streamlit presentation layer.

This module deliberately communicates only with the public FastAPI HTTP API.
It does not import backend services, database models, vector stores, or Ollama.
"""

from __future__ import annotations

from html import escape
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
from streamlit_app.components.status import get_backend_status, render_system_status
from streamlit_app.config import get_settings
from streamlit_app.evaluation import render_evaluation
from streamlit_app.styles import (
    apply_visual_system,
    render_pipeline_visual,
    render_sidebar_brand,
)
from streamlit_app.ui_helpers import (
    conversation_display_labels,
    friendly_api_error_message,
    page_display_data,
    set_comparison_pair,
    set_selected_document,
    upload_size_error,
)


PAGES = ("Home", "Documents", "Ask", "Analyze", "Compare", "Privacy", "Evaluation")
NAV_ITEMS = (
    ("Home", "Home", "home"),
    ("Documents", "Documents", "description"),
    ("Ask", "Ask AI", "chat"),
    ("Analyze", "Analyze", "auto_awesome"),
    ("Compare", "Compare", "compare_arrows"),
    ("Privacy", "Privacy", "shield"),
    ("Evaluation", "Evaluation", "analytics"),
)
INLINE_ICONS: dict[str, str] = {
    "upload": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M6 3.5h8l4 4V20.5H6z"/><path d="M14 3.5v4h4"/>'
        '<path d="M12 17V10M9 13l3-3 3 3"/></svg>'
    ),
    "file": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M6 3.5h8l4 4V20.5H6z"/><path d="M14 3.5v4h4"/>'
        '<path d="M9 12h6M9 15.5h6"/></svg>'
    ),
    "search": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="10.8" cy="10.8" r="6.2"/><path d="m16 16 4.5 4.5"/></svg>'
    ),
    "trophy": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M8 4h8v4.5a4 4 0 0 1-8 0z"/><path d="M8 6H4v1a4 4 0 0 0 4 4M16 6h4v1a4 4 0 0 1-4 4M12 12.5V17M8.5 20h7M10 17h4"/>'
        '</svg>'
    ),
    "shield": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M12 3.5 19 6v5.2c0 4.2-2.8 7.7-7 9.3-4.2-1.6-7-5.1-7-9.3V6z"/>'
        '<path d="m8.7 12.2 2.1 2.1 4.5-4.5"/></svg>'
    ),
    "chat": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M5 5.5h14v10H11l-4.5 3v-3H5z"/><path d="M8.5 9.5h7M8.5 12.5h4"/></svg>'
    ),
    "sparkle": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="m12 3 1.25 4.75L18 9l-4.75 1.25L12 15l-1.25-4.75L6 9l4.75-1.25z"/>'
        '<path d="m19 15 .65 2.35L22 18l-2.35.65L19 21l-.65-2.35L16 18l2.35-.65z"/></svg>'
    ),
    "conversation": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M4.5 5.5h11v8h-6l-3.5 2.5v-2.5h-1.5z"/>'
        '<path d="M9 18.5h5l3.5 2.5v-2.5H19.5v-8H18"/></svg>'
    ),
    "filter": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M4 6h16M7 12h10M10 18h4"/></svg>'
    ),
    "tag": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M4 5v6.2a2 2 0 0 0 .6 1.4l6.8 6.8a2 2 0 0 0 2.8 0l5.2-5.2a2 2 0 0 0 0-2.8L12.6 4.6A2 2 0 0 0 11.2 4H5a1 1 0 0 0-1 1z"/>'
        '<circle cx="8" cy="8" r="1"/></svg>'
    ),
    "table": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<rect x="4" y="4" width="16" height="16" rx="1.5"/><path d="M4 9h16M9 9v11M14 9v11"/></svg>'
    ),
    "compare": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M4 8h14M14 4l4 4-4 4M20 16H6M10 12l-4 4 4 4"/></svg>'
    ),
    "chart": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M5 20V10M12 20V4M19 20v-7"/><path d="M3 20h18"/></svg>'
    ),
}
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
        "comparison_pair": None,
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
        "home_document_count": None,
        "home_document_count_loaded": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("page") not in PAGES:
        st.session_state.page = "Home"


def main() -> None:
    st.set_page_config(page_title="DocuIntel", page_icon=":material/description:", layout="wide")
    initialize_state()
    apply_visual_system()
    client = get_api_client()

    page = render_sidebar(client)

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


def render_sidebar(client: ApiClient) -> str:
    """Render navigation and project branding without changing routing."""

    render_sidebar_brand()
    st.sidebar.markdown('<div class="di-sidebar-section-label">MAIN</div>', unsafe_allow_html=True)
    current_page = str(st.session_state.get("page", "Home"))
    for route, label, icon in NAV_ITEMS:
        st.sidebar.button(
            label,
            key=f"nav-{route.casefold()}",
            icon=f":material/{icon}:",
            type="primary" if route == current_page else "secondary",
            on_click=_select_page,
            args=(route,),
        )

    if st.session_state.backend_status is None:
        _refresh_backend_state(client)
    return str(st.session_state.page)


def _select_page(page: str) -> None:
    """Set the selected route from a sidebar navigation button."""

    st.session_state.page = page


def _refresh_backend_state(client: ApiClient) -> None:
    """Refresh status and allow the Home document KPI to reload from the API."""

    st.session_state.backend_status = get_backend_status(client)
    st.session_state.home_document_count_loaded = False


def render_home(client: ApiClient) -> None:
    st.markdown(
        '<div class="di-welcome-row"><span class="di-welcome-icon">✦</span>'
        '<span>Welcome back</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(
        key="home-hero-grid",
        horizontal=True,
        gap="large",
        vertical_alignment="center",
    ):
        with st.container(key="home-hero-copy", width="stretch"):
            st.markdown('<div class="di-eyebrow">AI-POWERED DOCUMENT INTELLIGENCE</div>', unsafe_allow_html=True)
            st.markdown(
                '<h1 class="di-hero-title">Turn documents into<br>'
                'actionable <span class="di-gradient-text">intelligence.</span></h1>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="di-hero-copy">Extract, understand, search, compare and protect documents '
                'using OCR, hybrid retrieval and grounded AI.</p>',
                unsafe_allow_html=True,
            )
            with st.container(horizontal=True, gap="small"):
                st.button(
                    "Upload Documents",
                    key="home-upload-documents",
                    icon=":material/upload_file:",
                    type="primary",
                    on_click=_select_page,
                    args=("Documents",),
                )
                st.button(
                    "Ask DocuIntel",
                    key="home-ask-docuintel",
                    icon=":material/auto_awesome:",
                    type="secondary",
                    on_click=_select_page,
                    args=("Ask",),
                )
        with st.container(key="home-hero-pipeline", width="stretch"):
            render_pipeline_visual()

    document_count = _load_home_document_count(client)
    kpis = (
        (
            "Documents",
            str(document_count) if document_count is not None else "Unavailable",
            "Persisted & indexed",
            "file",
            "violet",
        ),
        ("Hybrid Search", "Active", "Keyword + semantic retrieval", "search", "blue"),
        ("CrossEncoder Reranking", "Ready", "Improves relevance and precision", "trophy", "cyan"),
        ("Grounded Citations", "Enabled", "Answers backed by source evidence", "shield", "green"),
    )
    with st.container(
        key="home-kpi-grid",
        horizontal=True,
        gap="medium",
        vertical_alignment="top",
    ):
        for label, value, caption, icon, accent in kpis:
            with st.container(border=True, width="stretch", height="stretch"):
                if label == "Documents":
                    content = (
                        f'<div class="di-kpi-value">{value}</div>'
                        f'<div class="di-kpi-label">{label}</div>'
                        f'<div class="di-kpi-caption">{caption}</div>'
                    )
                else:
                    content = (
                        f'<div class="di-kpi-label">{label}</div>'
                        f'<div class="di-kpi-value">{value}</div>'
                        f'<div class="di-kpi-caption">{caption}</div>'
                    )
                st.markdown(
                    f'<div class="di-kpi-layout di-kpi-layout--{accent}">'
                    f'<div class="di-kpi-icon di-kpi-icon--{accent}">{INLINE_ICONS[icon]}</div>'
                    f'<div class="di-kpi-content">{content}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="di-section-heading"><h2>Key capabilities</h2></div>', unsafe_allow_html=True)
    capabilities = (
        ("Document Intelligence", "OCR, layout analysis and text extraction from PDFs.", "file", "violet"),
        ("Grounded RAG", "Ask questions with hybrid retrieval, reranking and source citations.", "chat", "blue"),
        ("Structured Analysis", "Extract fields, classify documents and query tables safely.", "table", "cyan"),
        ("Document Comparison", "Detect added, removed and modified content and table changes.", "compare", "green"),
        ("Privacy & Redaction", "Detect PII locally and create new redacted PDFs.", "shield", "violet"),
        ("Benchmark Evaluation", "Measure retrieval quality, RAG reliability and system performance.", "chart", "blue"),
    )
    with st.container(
        key="home-capabilities-grid",
        horizontal=True,
        gap="medium",
        vertical_alignment="top",
    ):
        for title, description, icon, accent in capabilities:
            with st.container(border=True, width="stretch", height="stretch"):
                st.markdown(
                    f'<div class="di-capability-layout">'
                    f'<div class="di-capability-icon di-capability-icon--{accent}">{INLINE_ICONS[icon]}</div>'
                    f'<div class="di-capability-content">'
                    f'<div class="di-capability-title">{title}</div>'
                    f'<div class="di-capability-description">{description}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )


def _load_home_document_count(client: ApiClient) -> int | None:
    """Load the real persisted document count once per UI session."""

    if st.session_state.get("home_document_count_loaded"):
        value = st.session_state.get("home_document_count")
        return int(value) if isinstance(value, int) else None
    try:
        response = documents_api.list_documents(client, page=1, page_size=1)
    except ApiError:
        count = None
    else:
        items = response.get("items")
        total = response.get("total")
        count = int(total) if isinstance(total, int) and not isinstance(total, bool) else None
        if count is None and isinstance(items, list):
            count = len(items)
    st.session_state.home_document_count = count
    st.session_state.home_document_count_loaded = True
    return count


def render_documents(client: ApiClient) -> None:
    with st.container(key="documents-page"):
        with st.container(
            key="documents-page-header",
            horizontal=True,
            gap="medium",
            vertical_alignment="center",
        ):
            st.markdown(
                '<div class="di-documents-header-copy">'
                '<div class="di-eyebrow">DOCUMENT LIBRARY</div>'
                '<h1 class="di-page-title">Documents</h1>'
                '<p class="di-page-subtitle">Upload, manage and inspect your indexed document library.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            refresh_clicked = st.button(
                "Refresh document list",
                key="documents-refresh",
                icon=":material/refresh:",
            )

        if refresh_clicked or st.session_state.documents is None:
            st.session_state.documents = documents_api.list_documents(client).get("items", [])

        with st.container(key="documents-upload-card", border=True):
            st.markdown(
                '<div class="di-card-heading">'
                f'<span class="di-card-heading-icon">{INLINE_ICONS["upload"]}</span>'
                '<span><strong>Upload documents</strong>'
                '<small>Add one or more PDF documents for extraction, indexing and analysis.</small></span>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.form("upload_pdf_form", clear_on_submit=True):
                st.markdown(
                    '<div class="di-upload-zone-copy">'
                    '<strong>PDF documents</strong>'
                    '<span>Choose one or more PDF files to add to your library.</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                uploaded_files = st.file_uploader(
                    "PDF documents",
                    type=["pdf"],
                    accept_multiple_files=True,
                    help="Select one or more PDFs. Each file is validated against the existing 25 MB backend limit.",
                )
                st.markdown(
                    '<div class="di-upload-hints"><span>Maximum 25 MB per document</span>'
                    '<span>Multiple files supported</span></div>',
                    unsafe_allow_html=True,
                )
                upload_clicked = st.form_submit_button(
                    "Upload selected PDFs",
                    type="primary",
                    icon=":material/upload_file:",
                )
        if upload_clicked:
            with st.container(key="documents-upload-results"):
                if not uploaded_files:
                    st.warning("Choose at least one PDF before uploading.")
                else:
                    upload_results: list[tuple[str, str, str]] = []
                    with st.spinner("Uploading and extracting selected PDFs..."):
                        for uploaded in uploaded_files:
                            content = uploaded.getvalue()
                            size_error = upload_size_error(
                                uploaded.name,
                                len(content),
                                get_settings().max_upload_size_mb,
                            )
                            if size_error:
                                upload_results.append((uploaded.name, "warning", size_error))
                                continue
                            try:
                                response = documents_api.upload_document(client, uploaded.name, content)
                            except ApiError as exc:
                                result_kind = "warning" if exc.status_code == 409 else "error"
                                upload_results.append(
                                    (uploaded.name, result_kind, friendly_api_error_message(exc))
                                )
                            else:
                                display_name = response.get("original_filename", uploaded.name)
                                upload_results.append((uploaded.name, "success", f"{display_name} — uploaded"))
                    for filename, result_kind, message in upload_results:
                        if result_kind == "success":
                            st.success(message)
                        elif result_kind == "warning":
                            st.warning(f"{filename} — {message}")
                        else:
                            st.error(f"{filename} — failed: {message}")
                    st.session_state.documents = documents_api.list_documents(client).get("items", [])

        items = st.session_state.documents or []
        st.markdown(
            '<div class="di-documents-section-heading">'
            '<h2>Document library</h2>'
            '<p>Select an indexed document to inspect its processing results.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if not items:
            with st.container(key="documents-empty-state", border=True):
                st.info("No documents have been uploaded yet.", icon=":material/inventory_2:")
            return

        labels = {str(item["id"]): _document_label(item) for item in items}
        ids = list(labels)
        current = st.session_state.selected_document_id
        index = ids.index(current) if current in ids else 0
        with st.container(key="documents-library-card", border=True):
            selected_id = st.selectbox("Select a document", ids, index=index, format_func=labels.get)
        set_selected_document(st.session_state, selected_id)
        render_document_detail(client, selected_id)


def render_document_detail(client: ApiClient, document_id: str) -> None:
    detail = documents_api.get_document(client, document_id)
    filename = escape(str(detail.get("original_filename", "Document detail")))
    status = escape(str(detail.get("status", "unknown")))
    indexed = "Yes" if detail.get("is_indexed") else "No"
    with st.container(key="document-overview", border=True):
        st.markdown(
            '<div class="di-selected-document-heading">'
            '<div class="di-eyebrow">SELECTED DOCUMENT</div>'
            f'<h2>{filename}</h2>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container(
            key="document-overview-kpis",
            horizontal=True,
            gap="medium",
            vertical_alignment="top",
        ):
            metrics = (
                ("Status", f'<span class="di-status-badge">{status}</span>'),
                ("Pages", escape(str(detail.get("page_count", "—")))),
                ("Chunks", escape(str(detail.get("chunk_count", "—")))),
                ("Indexed", indexed),
            )
            for label, value in metrics:
                with st.container(width="stretch"):
                    st.markdown(
                        f'<div class="di-document-metric">'
                        f'<span class="di-document-metric-label">{label}</span>'
                        f'<span class="di-document-metric-value">{value}</span></div>',
                        unsafe_allow_html=True,
                    )

        metadata = detail.get("metadata") or {}
        summary = detail.get("summary") or {}
        with st.expander("Metadata and processing summary"):
            st.json({"metadata": metadata, "summary": summary})

    with st.container(
        key="document-actions",
        horizontal=True,
        gap="small",
        vertical_alignment="bottom",
    ):
        if st.button("Rebuild index", key=f"index-{document_id}", icon=":material/refresh:"):
            with st.spinner("Chunking and embedding document..."):
                response = documents_api.index_document(client, document_id)
            st.success(f"Indexed {response.get('chunks_created', 0)} chunks.")
            st.session_state.documents = documents_api.list_documents(client).get("items", [])
        with st.form(f"delete-document-{document_id}"):
            confirmed = st.checkbox(
                "I understand this deletes the document",
                key=f"confirm-delete-{document_id}",
            )
            delete_clicked = st.form_submit_button("Delete document", icon=":material/delete:")
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
            st.info("No extracted content is available for this document.")
        for chunk in chunks:
            sequence = escape(str(chunk.get("sequence_number", "—")))
            start_page = escape(str(chunk.get("start_page", "—")))
            end_page = escape(str(chunk.get("end_page", "—")))
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            chunk_meta = [
                f"Content type: {escape(str(chunk.get('content_type', 'unknown')))}",
                f"Characters: {escape(str(chunk.get('character_count', '—')))}",
            ]
            if chunk_id is not None:
                chunk_meta.insert(0, f"Chunk ID: {escape(str(chunk_id))}")
            with st.expander(
                f"Chunk {sequence} · pages {start_page}–{end_page}"
            ):
                st.markdown(
                    f'<div class="di-chunk-meta">{" · ".join(chunk_meta)}</div>',
                    unsafe_allow_html=True,
                )
                st.write(chunk.get("text") or "")
                st.caption("Extracted chunk text")


def render_analyze(client: ApiClient) -> None:
    """Render transient summary and classification controls for one document."""

    with st.container(key="analyze-page"):
        with st.container(
            key="analyze-page-header",
            horizontal=True,
            gap="medium",
            vertical_alignment="center",
        ):
            with st.container(key="analyze-page-header-copy", width="stretch"):
                st.markdown(
                    '<div class="di-eyebrow">DOCUMENT INTELLIGENCE WORKSPACE</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<h1 class="di-page-title">Analyze</h1>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="di-page-subtitle">Summarize, classify, extract structured information and '
                    'query document tables with grounded evidence.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="di-analyze-header-icon">{INLINE_ICONS["chart"]}</div>',
                unsafe_allow_html=True,
            )

        documents = _load_documents_for_filters(client)
        if not documents:
            with st.container(key="analyze-empty-state", border=True):
                st.info("No documents are currently visible through the API.")
            return

        labels = {str(item["id"]): _document_label(item) for item in documents}
        document_ids = list(labels)
        current = st.session_state.selected_document_id
        index = document_ids.index(current) if current in document_ids else 0
        with st.container(key="analyze-document-context", border=True):
            selected_id = st.selectbox(
                "Select a document",
                document_ids,
                index=index,
                format_func=labels.get,
                key="analyze-document",
            )
            set_selected_document(st.session_state, selected_id)
            detail = documents_api.get_document(client, selected_id)
            selected_item = next((item for item in documents if str(item.get("id")) == selected_id), {})
            filename = str(detail.get("original_filename") or selected_item.get("original_filename") or "Unnamed document")
            status = str(detail.get("status") or selected_item.get("status") or "unknown")
            indexed = detail.get("is_indexed") is True
            st.markdown(
                f'<div class="di-selected-document-heading"><div class="di-eyebrow">SELECTED DOCUMENT</div>'
                f'<h2>{escape(filename)}</h2></div>',
                unsafe_allow_html=True,
            )
            with st.container(
                key="analyze-document-kpis",
                horizontal=True,
                gap="medium",
                vertical_alignment="top",
            ):
                metrics = (
                    ("Status", status, "violet"),
                    ("Pages", detail.get("page_count", "—"), "blue"),
                    ("Chunks", detail.get("chunk_count", "—"), "cyan"),
                    ("Indexed", "Yes" if indexed else "No", "green"),
                )
                for label, value, accent in metrics:
                    st.markdown(
                        f'<div class="di-analyze-metric di-analyze-metric--{accent}">'
                        f'<span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>',
                        unsafe_allow_html=True,
                    )

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
            with st.container(key="analyze-summary-card", border=True):
                st.markdown(
                    f'<div class="di-analyze-card-heading"><span class="di-analyze-card-icon">'
                    f'{INLINE_ICONS["file"]}</span><div><h2>Document summary</h2>'
                    '<p>Generate a grounded summary using evidence from the selected indexed document.</p>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
                with st.form("document-summary-form", border=False):
                    style = st.selectbox(
                        "Summary style",
                        ["brief", "detailed", "bullet_points"],
                        format_func=lambda value: value.replace("_", " ").title(),
                    )
                    summary_clicked = st.form_submit_button(
                        "Generate summary",
                        type="primary",
                        icon=":material/auto_awesome:",
                    )
                if summary_clicked:
                    with st.spinner("Generating grounded document summary..."):
                        st.session_state.last_summary = analysis_api.summarize(
                            client,
                            selected_id,
                            style=style,
                        )
                summary = st.session_state.last_summary
                if summary and str(summary.get("document_id")) == selected_id:
                    with st.container(key="analyze-summary-result", border=True):
                        st.markdown('<div class="di-result-kicker">GROUNDED SUMMARY</div>', unsafe_allow_html=True)
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
            with st.container(key="analyze-classification-card", border=True):
                st.markdown(
                    f'<div class="di-analyze-card-heading"><span class="di-analyze-card-icon di-analyze-card-icon--blue">'
                    f'{INLINE_ICONS["tag"]}</span><div><h2>Document classification</h2>'
                    '<p>Classify the selected document using only the labels you provide.</p>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="di-form-section-label">Allowed labels</div>', unsafe_allow_html=True)
                st.caption("Enter at least two allowed labels, one per line. Labels are not persisted.")
                with st.form("document-classification-form", border=False):
                    label_text = st.text_area(
                        "Allowed labels",
                        value="Employment Policy\nExpense Policy\nOther",
                        height=120,
                    )
                    classify_clicked = st.form_submit_button(
                        "Classify document",
                        type="primary",
                        icon=":material/label:",
                    )
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
                    with st.container(key="analyze-classification-result", border=True):
                        st.markdown('<div class="di-result-kicker">CLASSIFICATION RESULT</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="di-classification-label">'
                            f'{escape(str(classification.get("selected_label", "—")))}</div>',
                            unsafe_allow_html=True,
                        )
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
    """Render the HTTP-only document change-intelligence workspace."""

    with st.container(key="compare-page"):
        with st.container(
            key="compare-page-header",
            horizontal=True,
            gap="medium",
            vertical_alignment="center",
        ):
            with st.container(key="compare-page-header-copy", width="stretch"):
                st.markdown(
                    '<div class="di-eyebrow">DOCUMENT CHANGE INTELLIGENCE</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<h1 class="di-page-title">Compare</h1>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="di-page-subtitle">Detect added, removed and modified content across document '
                    'versions with source-grounded evidence.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="di-compare-header-icon">{INLINE_ICONS["compare"]}</div>',
                unsafe_allow_html=True,
            )

        documents = [
            item
            for item in _load_documents_for_filters(client)
            if item.get("status") == "ready" and item.get("is_indexed") is True
        ]
        if len(documents) < 2:
            with st.container(key="compare-empty-state", border=True):
                st.markdown(
                    '<div class="di-compare-empty-heading">Comparison workspace</div>',
                    unsafe_allow_html=True,
                )
                st.info("At least two ready indexed documents are required for comparison.")
            return

        labels = {str(item["id"]): _document_label(item) for item in documents}
        document_ids = list(labels)
        with st.container(key="compare-setup-card", border=True):
            st.markdown(
                f'<div class="di-compare-card-heading"><span class="di-compare-card-icon">'
                f'{INLINE_ICONS["compare"]}</span><div><h2>Set up a comparison</h2>'
                '<p>Choose the original and updated evidence you want to inspect.</p></div></div>',
                unsafe_allow_html=True,
            )
            with st.form("comparison-form", border=False):
                base_column, target_column = st.columns(2, gap="large")
                with base_column:
                    st.markdown(
                        '<div class="di-compare-side-label"><span>BASE / ORIGINAL</span>'
                        '<small>Earlier or reference document</small></div>',
                        unsafe_allow_html=True,
                    )
                    base_id = st.selectbox(
                        "Base document",
                        document_ids,
                        format_func=labels.get,
                        key="comparison-base-document",
                    )
                with target_column:
                    st.markdown(
                        '<div class="di-compare-side-label di-compare-side-label--target">'
                        '<span>TARGET / UPDATED</span><small>Later or changed document</small></div>',
                        unsafe_allow_html=True,
                    )
                    target_id = st.selectbox(
                        "Target document",
                        document_ids,
                        format_func=labels.get,
                        key="comparison-target-document",
                    )
                st.markdown(
                    '<div class="di-compare-direction-note"><span>Version comparison</span> treats Base as '
                    'the older document and Target as the newer document. Document comparison keeps the selected '
                    'pair directional without inferring chronology.</div>',
                    unsafe_allow_html=True,
                )
                with st.container(
                    key="compare-options",
                    horizontal=True,
                    gap="medium",
                    vertical_alignment="bottom",
                ):
                    mode = st.selectbox(
                        "Comparison mode",
                        ["document", "version"],
                        format_func=lambda value: "Version comparison" if value == "version" else "Document comparison",
                        key="comparison-mode",
                    )
                    include_tables = st.checkbox(
                        "Compare structured tables",
                        value=True,
                        key="comparison-tables",
                    )
                    include_unchanged = st.checkbox(
                        "Include unchanged items",
                        value=False,
                        key="comparison-unchanged",
                    )
                    generate_summary = st.checkbox(
                        "Generate grounded change summary",
                        value=True,
                        key="comparison-summary",
                    )
                compare_clicked = st.form_submit_button(
                    "Compare documents",
                    type="primary",
                    icon=":material/compare_arrows:",
                )

        set_comparison_pair(st.session_state, base_id, target_id)
        if compare_clicked:
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

        base_document = response.get("base_document") or {}
        target_document = response.get("target_document") or {}
        base_filename = str(base_document.get("filename") or labels.get(base_id) or "Base document")
        target_filename = str(target_document.get("filename") or labels.get(target_id) or "Target document")
        statistics = response.get("statistics") if isinstance(response.get("statistics"), dict) else {}
        changes = response.get("changes") if isinstance(response.get("changes"), list) else []

        with st.container(key="compare-result", border=True):
            st.markdown(
                '<div class="di-compare-result-kicker">COMPARISON RESULT</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="compare-result-pair", horizontal=True, gap="small", vertical_alignment="center"):
                st.markdown(
                    f'<div class="di-compare-document-chip di-compare-document-chip--base">'
                    f'<span>BASE / ORIGINAL</span><strong>{escape(base_filename)}</strong></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="di-compare-pair-divider">vs</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="di-compare-document-chip di-compare-document-chip--target">'
                    f'<span>TARGET / UPDATED</span><strong>{escape(target_filename)}</strong></div>',
                    unsafe_allow_html=True,
                )

            stat_specs = (
                ("added_count", "Added", "green", "+"),
                ("removed_count", "Removed", "red", "−"),
                ("modified_count", "Modified", "indigo", "~"),
                ("unchanged_count", "Unchanged", "neutral", "="),
                ("table_change_count", "Table changes", "cyan", "▦"),
            )
            with st.container(
                key="compare-stats-grid",
                horizontal=True,
                gap="small",
                vertical_alignment="top",
            ):
                for key, label, accent, symbol in stat_specs:
                    value = escape(str(statistics.get(key, 0)))
                    with st.container(key=f"compare-stat-{key}", border=True, width="stretch"):
                        st.markdown(
                            f'<div class="di-compare-stat di-compare-stat--{accent}">'
                            f'<span class="di-compare-stat-symbol">{symbol}</span>'
                            f'<div><strong>{value}</strong><small>{escape(label)}</small></div></div>',
                            unsafe_allow_html=True,
                        )

        with st.container(key="compare-summary-card", border=True):
            st.markdown(
                '<div class="di-compare-section-heading"><span class="di-compare-section-icon">'
                f'{INLINE_ICONS["sparkle"]}</span><div><div class="di-compare-section-kicker">CHANGE SUMMARY</div>'
                '<h2>Grounded change summary</h2><p>The existing comparison summary is shown with its source labels intact.</p>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(response.get("summary") or "No change summary was returned.")

        change_groups = (
            ("modified", "Modified", "Differences found in paired content.", "indigo", True),
            ("added", "Added", "Content present in Target but not Base.", "green", True),
            ("removed", "Removed", "Content present in Base but not Target.", "red", True),
            ("unchanged", "Unchanged", "Content retained across both documents.", "neutral", False),
        )
        for change_type, title, description, accent, expand_first in change_groups:
            matching = [change for change in changes if isinstance(change, dict) and change.get("change_type") == change_type]
            with st.container(key=f"compare-section-{change_type}", border=True):
                st.markdown(
                    f'<div class="di-compare-section-heading di-compare-section-heading--{accent}">'
                    f'<span class="di-compare-section-icon">{_comparison_group_symbol(change_type)}</span>'
                    f'<div><div class="di-compare-section-kicker">{escape(title.upper())}</div>'
                    f'<h2>{escape(title)} <em>{len(matching)}</em></h2>'
                    f'<p>{escape(description)}</p></div></div>',
                    unsafe_allow_html=True,
                )
                if not matching:
                    empty_copy = (
                        "Unchanged items are hidden because the option is disabled."
                        if change_type == "unchanged" and not include_unchanged
                        else f"No {title.lower()} changes were detected for this pair."
                    )
                    st.caption(empty_copy)
                    continue
                for index, change in enumerate(matching, start=1):
                    change_id = str(change.get("change_id") or f"Change {index}")
                    scope = str(change.get("scope") or "text").replace("_", " ").title()
                    with st.expander(
                        f"{change_id} · {scope}",
                        expanded=expand_first and index == 1,
                    ):
                        st.markdown(
                            f'<div class="di-compare-change-meta"><span>{escape(change_id)}</span>'
                            f'<small>{escape(scope)} change</small></div>',
                            unsafe_allow_html=True,
                        )
                        with st.container(key=f"compare-change-body-{change_id}"):
                            before, after = st.columns(2, gap="medium")
                            with before:
                                st.markdown('<div class="di-compare-side-heading di-compare-side-heading--base">BASE CONTENT</div>', unsafe_allow_html=True)
                                _render_comparison_text(change.get("base_text"), empty_label="Not present in Base")
                                st.markdown('<div class="di-compare-evidence-label">BASE EVIDENCE</div>', unsafe_allow_html=True)
                                _render_comparison_provenance(change.get("base_provenance"), "Base")
                            with after:
                                st.markdown('<div class="di-compare-side-heading di-compare-side-heading--target">TARGET CONTENT</div>', unsafe_allow_html=True)
                                _render_comparison_text(change.get("target_text"), empty_label="Not present in Target")
                                st.markdown('<div class="di-compare-evidence-label">TARGET EVIDENCE</div>', unsafe_allow_html=True)
                                _render_comparison_provenance(change.get("target_provenance"), "Target")
                        _render_table_change_detail(change)

        if not any(isinstance(change, dict) and change.get("scope") == "table" for change in changes):
            with st.container(key="compare-table-empty-state", border=True):
                st.markdown(
                    '<div class="di-compare-table-empty-heading">Table intelligence</div>',
                    unsafe_allow_html=True,
                )
                st.caption("No structured table changes were detected for this comparison.")

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


def _comparison_group_symbol(change_type: str) -> str:
    """Return a compact semantic symbol for a comparison change group."""

    return {
        "modified": "~",
        "added": "+",
        "removed": "−",
        "unchanged": "=",
    }.get(change_type, "•")


def _render_comparison_text(value: Any, *, empty_label: str) -> None:
    """Render comparison content as escaped, line-preserving text."""

    is_empty = value is None or not str(value).strip()
    text = escape(empty_label if is_empty else str(value)).replace("\n", "<br>")
    modifier = " di-compare-content-text--empty" if is_empty else ""
    st.markdown(
        f'<div class="di-compare-content-text{modifier}">{text}</div>',
        unsafe_allow_html=True,
    )


def _comparison_page_label(evidence: dict[str, Any]) -> str:
    """Return the most specific available page label without inventing metadata."""

    if evidence.get("page_number") is not None:
        return str(evidence["page_number"])
    start_page = evidence.get("start_page")
    end_page = evidence.get("end_page")
    if start_page is None:
        return "—"
    if end_page is not None and end_page != start_page:
        return f"{start_page}–{end_page}"
    return str(start_page)


def _render_comparison_provenance(items: Any, side_label: str) -> None:
    """Render source labels and available provenance for one comparison side."""

    evidence_items = items if isinstance(items, list) else []
    valid_items = [item for item in evidence_items if isinstance(item, dict)]
    if not valid_items:
        st.caption(f"No {side_label.lower()} source evidence available.")
        return
    for evidence in valid_items:
        source_id = escape(str(evidence.get("source_id") or "Source"))
        filename = escape(str(evidence.get("filename") or "Document"))
        page = escape(_comparison_page_label(evidence))
        details = [f"Page {page}"]
        if evidence.get("chunk_id"):
            details.append(f"Chunk {escape(str(evidence['chunk_id']))}")
        if evidence.get("section_heading"):
            details.append(escape(str(evidence["section_heading"])))
        st.markdown(
            f'<div class="di-compare-source"><span class="di-compare-source-badge">{source_id}</span>'
            f'<div><strong>{filename}</strong><small>{" · ".join(details)}</small></div></div>',
            unsafe_allow_html=True,
        )


def _render_table_change_detail(change: dict[str, Any]) -> None:
    """Render structured table metadata while retaining the original JSON detail."""

    detail = change.get("table_detail")
    if not isinstance(detail, dict):
        return
    change_kind = str(detail.get("table_change_type") or "table change").replace("_", " ").title()
    row_key = detail.get("row_key") or "—"
    column = detail.get("column") or "—"
    before = detail.get("before") or "—"
    after = detail.get("after") or "—"
    change_id = escape(str(change.get("change_id") or "table-change"))
    with st.container(key=f"compare-table-detail-{change_id}", border=True):
        st.markdown(
            f'<div class="di-table-change-heading"><span>TABLE CHANGE</span><strong>{escape(change_kind)}</strong></div>',
            unsafe_allow_html=True,
        )
        with st.container(key=f"compare-table-detail-grid-{change_id}", horizontal=True, gap="small"):
            for label, value in (("Column", column), ("Row / key", row_key), ("Before", before), ("After", after)):
                st.markdown(
                    f'<div class="di-table-change-cell"><small>{escape(label)}</small>'
                    f'<strong>{escape(str(value))}</strong></div>',
                    unsafe_allow_html=True,
                )
        row_values = detail.get("row_values")
        if isinstance(row_values, dict) and row_values:
            st.markdown('<div class="di-table-change-row-label">Row values</div>', unsafe_allow_html=True)
            st.json(row_values)
        with st.expander("Technical table details", icon=":material/data_object:"):
            st.json(detail)


def render_privacy(client: ApiClient) -> None:
    """Render the review-first, explicit-selection privacy workflow."""

    with st.container(key="privacy-page"):
        with st.container(
            key="privacy-page-header",
            horizontal=True,
            gap="medium",
            vertical_alignment="center",
        ):
            with st.container(key="privacy-page-header-copy", width="stretch"):
                st.markdown(
                    '<div class="di-eyebrow">PRIVACY &amp; DOCUMENT PROTECTION</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<h1 class="di-page-title">Privacy &amp; Redaction</h1>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="di-page-subtitle">Detect sensitive information, review each finding and create '
                    'selectively redacted PDF copies.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="di-privacy-header-icon">{INLINE_ICONS["shield"]}</div>',
                unsafe_allow_html=True,
            )

        documents = [
            item
            for item in _load_documents_for_filters(client)
            if item.get("status") == "ready"
        ]
        if not documents:
            with st.container(key="privacy-empty-state", border=True):
                st.markdown(
                    f'<div class="di-privacy-empty-icon">{INLINE_ICONS["shield"]}</div>'
                    '<div class="di-privacy-empty-heading">No ready documents available</div>'
                    '<p class="di-privacy-empty-copy">Upload and process a document before starting a privacy scan.</p>',
                    unsafe_allow_html=True,
                )
            return

        labels = {str(item["id"]): _document_label(item) for item in documents}
        document_ids = list(labels)
        with st.container(key="privacy-scan-card", border=True):
            st.markdown(
                f'<div class="di-privacy-card-heading"><span class="di-privacy-card-icon">'
                f'{INLINE_ICONS["shield"]}</span><div><h2>Scan for sensitive information</h2>'
                '<p>Select a document and choose the PII types you want DocuIntel to detect.</p></div></div>',
                unsafe_allow_html=True,
            )
            with st.form("privacy-scan-form", border=False):
                with st.container(
                    key="privacy-scan-fields",
                    horizontal=True,
                    gap="large",
                    vertical_alignment="bottom",
                ):
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
                st.markdown(
                    '<div class="di-privacy-form-note"><strong>Review first.</strong> Scanning only reports '
                    'validated findings; it never changes the original document.</div>',
                    unsafe_allow_html=True,
                )
                scan_clicked = st.form_submit_button(
                    "Scan for PII",
                    type="primary",
                    icon=":material/search:",
                )

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

        counts = scan.get("counts_by_type") if isinstance(scan.get("counts_by_type"), dict) else {}
        with st.container(key="privacy-overview-card", border=True):
            st.markdown(
                '<div class="di-privacy-section-kicker">SCAN RESULTS</div>'
                '<div class="di-privacy-section-heading"><h2>Detection overview</h2>'
                '<p>Only supported, high-confidence structured PII returned by the detector is shown here.</p></div>',
                unsafe_allow_html=True,
            )
            with st.container(
                key="privacy-overview-grid",
                horizontal=True,
                gap="medium",
                vertical_alignment="top",
            ):
                metric_specs = (
                    ("Total detections", scan.get("detection_count", "—"), "indigo"),
                    ("Email", counts.get("email", "—"), "violet"),
                    ("Phone number", counts.get("phone_number", "—"), "blue"),
                    ("IBAN", counts.get("iban", "—"), "cyan"),
                    ("Credit card", counts.get("credit_card", "—"), "green"),
                )
                for label, value, accent in metric_specs:
                    st.markdown(
                        f'<div class="di-privacy-metric di-privacy-metric--{accent}">'
                        f'<span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>',
                        unsafe_allow_html=True,
                    )
            if scan.get("total_time_ms") is not None:
                scan_time_ms = float(scan["total_time_ms"])
                st.caption(f"Scan completed in {scan_time_ms:.1f} ms")

        detections = scan.get("detections") if isinstance(scan.get("detections"), list) else []
        if not detections:
            with st.container(key="privacy-no-pii-state", border=True):
                st.markdown(
                    '<div class="di-privacy-empty-heading">No supported PII detected</div>'
                    '<p class="di-privacy-empty-copy">No high-confidence email, phone number, IBAN or '
                    'credit-card values were found for the selected scan.</p>',
                    unsafe_allow_html=True,
                )
            return

        redactable_count = sum(bool(detection.get("redactable")) for detection in detections)
        with st.container(key="privacy-review-card", border=True):
            st.markdown(
                '<div class="di-privacy-section-kicker">REVIEW BEFORE REMOVAL</div>'
                '<div class="di-privacy-section-heading"><h2>Review detections</h2>'
                '<p>Choose only the findings you explicitly want removed. Scanning never redacts automatically.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if not redactable_count:
                st.warning(
                    "Detections were found, but none have verified page coordinates for safe redaction.",
                    icon=":material/warning:",
                )
            with st.form("privacy-redaction-form", border=False):
                selected_ids: list[str] = []
                for index, detection in enumerate(detections):
                    detection_id = str(detection.get("detection_id", ""))
                    redactable = bool(detection.get("redactable"))
                    pii_type = _privacy_type_label(detection.get("pii_type"))
                    page_number = detection.get("page_number")
                    page_label = f"Page {page_number}" if page_number is not None else "Page not provided"
                    matched_text = str(detection.get("matched_text") or "Value not returned")
                    safe_detection_key = re.sub(r"[^A-Za-z0-9_-]", "-", detection_id) or str(index)
                    with st.container(
                        key=f"privacy-detection-{index}-{safe_detection_key}",
                        border=True,
                    ):
                        with st.container(
                            key=f"privacy-detection-content-{index}",
                            horizontal=True,
                            gap="medium",
                            vertical_alignment="center",
                        ):
                            st.markdown(
                                f'<div class="di-privacy-detection-copy"><div class="di-privacy-type">'
                                f'{escape(pii_type)}</div><strong>{escape(matched_text)}</strong>'
                                f'<span>{escape(page_label)}</span></div>',
                                unsafe_allow_html=True,
                            )
                            status_label = "Safe to redact" if redactable else "Review only"
                            status_class = "eligible" if redactable else "unavailable"
                            st.markdown(
                                f'<div class="di-privacy-detection-status di-privacy-detection-status--{status_class}">'
                                f'{escape(status_label)}</div>',
                                unsafe_allow_html=True,
                            )
                            if st.checkbox(
                                "Select for redaction" if redactable else "Not safely redactable",
                                key=f"privacy-select-{detection_id}",
                                disabled=not redactable,
                            ):
                                selected_ids.append(detection_id)

                with st.container(key="privacy-redaction-action", border=True):
                    st.markdown(
                        '<div class="di-privacy-action-heading"><span class="di-privacy-action-icon">'
                        f'{INLINE_ICONS["file"]}</span><div><h3>Create redacted copy</h3>'
                        '<p>Only selected findings will be removed. The original PDF remains unchanged.</p>'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )
                    redact_clicked = st.form_submit_button(
                        "Create redacted PDF",
                        type="primary",
                        icon=":material/cleaning_services:",
                    )
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
            artifact = redaction.get("artifact") if isinstance(redaction.get("artifact"), dict) else {}
            with st.container(key="privacy-artifact-card", border=True):
                st.markdown(
                    '<div class="di-privacy-artifact-kicker">DERIVATIVE ARTIFACT</div>'
                    '<div class="di-privacy-artifact-heading"><span class="di-privacy-artifact-icon">'
                    f'{INLINE_ICONS["shield"]}</span><div><h2>Redacted PDF created</h2>'
                    f'<p>{escape(str(redaction.get("redacted_count", 0)))} selected finding(s) removed. '
                    'The original document is unchanged.</p></div></div>',
                    unsafe_allow_html=True,
                )
                artifact_filename = str(artifact.get("filename") or "redacted-document.pdf")
                download_url = str(artifact.get("download_url") or "")
                if download_url:
                    try:
                        artifact_bytes = privacy_api.download_redacted_artifact(client, download_url)
                        st.download_button(
                            "Download redacted PDF",
                            data=artifact_bytes,
                            file_name=artifact_filename,
                            mime="application/pdf",
                            key=f"privacy-download-{artifact.get('artifact_id', 'artifact')}",
                        )
                    except ApiError as exc:
                        st.error(format_module_12_4_api_error(exc, action="download"))
                else:
                    st.warning("The redacted artifact did not include a download URL.")
            render_timings(
                redaction,
                ("redaction_time_ms", "total_time_ms"),
            )


def _privacy_type_label(value: Any) -> str:
    """Format a backend PII enum for presentation without changing its value."""

    return str(value or "PII").replace("_", " ").upper()


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

    with st.container(key="analyze-extraction-card", border=True):
        st.markdown(
            f'<div class="di-analyze-card-heading"><span class="di-analyze-card-icon di-analyze-card-icon--cyan">'
            f'{INLINE_ICONS["file"]}</span><div><h2>Structured extraction</h2>'
            '<p>Extract bounded typed fields from evidence in the selected document.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if not indexed_documents or selected_id not in {str(item["id"]) for item in indexed_documents}:
            st.info("Select an indexed document to use structured extraction.")
            return
        st.markdown('<div class="di-form-section-label">Requested fields</div>', unsafe_allow_html=True)
        with st.form("structured-extraction-form", border=False):
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
            for field_index, field in enumerate(response.get("fields", []), start=1):
                status = field.get("status", "not_found")
                value = field.get("value")
                status_label = {
                    "found": "FOUND",
                    "not_found": "NOT FOUND",
                }.get(str(status), str(status).upper())
                status_class = "found" if status == "found" else "not-found"
                field_name = str(field.get("field", "field"))
                with st.container(key=f"analyze-field-result-{selected_id}-{field_index}", border=True):
                    st.markdown(
                        f'<div class="di-extraction-field-heading"><div><div class="di-result-kicker">FIELD NAME</div>'
                        f'<strong>{escape(field_name)}</strong></div>'
                        f'<span class="di-field-status di-field-status--{status_class}">{escape(status_label)}</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="di-field-value-label">Value</div>', unsafe_allow_html=True)
                    st.write(value if value is not None else "Not found in the supplied evidence.")
                    if field.get("candidates"):
                        with st.expander("Alternative candidates"):
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

    with st.container(key="analyze-tables-card", border=True):
        st.markdown(
            f'<div class="di-analyze-card-heading"><span class="di-analyze-card-icon di-analyze-card-icon--green">'
            f'{INLINE_ICONS["table"]}</span><div><h2>Table intelligence</h2>'
            '<p>Inspect extracted tables and run deterministic, source-mapped table operations.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
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
            with st.container(key="analyze-table-empty-state", border=True):
                st.markdown('<div class="di-table-empty-heading">No structured tables detected</div>', unsafe_allow_html=True)
                st.info(empty_message)
                st.caption("This document does not contain a structured table available for table analysis.")
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
            st.markdown('<div class="di-table-preview-label">Table preview</div>', unsafe_allow_html=True)
            st.table([dict(zip(headers, row, strict=False)) for row in rows])
        if preview.get("truncated"):
            st.caption("Preview truncated; supported operations use the bounded backend table data.")
        with st.form(f"table-query-form-{selected_id}", border=False):
            st.markdown(
                '<div class="di-table-examples"><strong>Examples:</strong> Which product has the highest quantity? '
                'What is the total price? How many rows are there?</div>',
                unsafe_allow_html=True,
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
            with st.container(key="analyze-table-result", border=True):
                st.markdown('<div class="di-result-kicker">TABLE ANSWER</div>', unsafe_allow_html=True)
                st.markdown(response.get("answer") or "No table answer was returned.")
                st.markdown('<div class="di-table-result-label">Deterministic operation result</div>', unsafe_allow_html=True)
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
    st.markdown(
        '<div class="di-analysis-evidence-heading"><span class="di-analyze-card-icon">'
        f'{INLINE_ICONS["file"]}</span><div><h3>Evidence &amp; sources</h3>'
        '<p>Grounding metadata returned for this analysis.</p></div></div>',
        unsafe_allow_html=True,
    )
    for index, source in enumerate(sources, start=1):
        source_id = str(source.get("source_id") or f"Source {index}")
        filename = str(source.get("filename") or "document")
        page = source.get("start_page", "—")
        end_page = source.get("end_page")
        if end_page is not None and end_page != page:
            page = f"{page}–{end_page}"
        with st.expander(f"{source_id} · {filename} · page {page}"):
            st.markdown(
                f'<div class="di-analysis-source-heading"><span class="di-source-badge">'
                f'{escape(source_id)}</span><div><strong>{escape(filename)}</strong>'
                f'<span>Page {escape(str(page))}</span></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Chunk sequence {source.get('sequence_number', '—')} · "
                f"{source.get('section_heading') or 'No section heading'}"
            )
            st.markdown('<div class="di-evidence-label">Evidence</div>', unsafe_allow_html=True)
            st.write(source.get("excerpt") or "")


def render_pages(client: ApiClient, document_id: str, pages: list[dict[str, Any]]) -> None:
    if not pages:
        st.info("No extracted content is available for this document.")
        return
    page_numbers = [int(page["page_number"]) for page in pages]
    page_number = st.selectbox("Page", page_numbers, key=f"page-select-{document_id}")
    page = documents_api.get_page(client, document_id, page_number)
    display = page_display_data(page)
    st.markdown(
        f'<div class="di-page-view-heading"><strong>Page {page_number}</strong>'
        f'<span>Extraction method: {escape(str(display["extraction_method"]))}</span></div>',
        unsafe_allow_html=True,
    )
    if display["is_ocr"]:
        st.caption(str(display["quality_note"]))
        with st.expander(str(display["expander_label"])):
            st.text(display["text"])
    else:
        st.markdown('<div class="di-page-extracted-label">Extracted text</div>', unsafe_allow_html=True)
        st.write(display["text"])
    if page.get("layout_elements") or page.get("tables"):
        with st.expander("Structured page data"):
            st.json({"layout_elements": page.get("layout_elements", []), "tables": page.get("tables", [])})


def render_search(client: ApiClient, *, nested: bool = False) -> None:
    if nested:
        heading = "Search document evidence"
        copy = "Inspect the chunks returned by keyword, semantic or hybrid retrieval before answer generation."
        card_key = "ask-search-card"
    else:
        heading = "Search evidence"
        copy = "Use Module 5 retrieval and optionally Module 6 reranking."
        card_key = "search-card"

    st.markdown(
        f'<div class="di-ask-section-heading"><span class="di-ask-section-icon">'
        f'{INLINE_ICONS["search"]}</span><div><h2>{heading}</h2>'
        f'<p>{copy}</p></div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key=card_key, border=True):
        documents = _load_documents_for_filters(client)
        scope_mode = render_document_scope_selector(key_prefix="search")
        with st.form("search_form", border=False):
            st.markdown('<div class="di-form-section-label">Search controls</div>', unsafe_allow_html=True)
            query = st.text_input("Search query", placeholder="Find a policy, clause or document detail")
            with st.container(key="search-retrieval-settings", horizontal=True, gap="medium"):
                mode = st.selectbox("Search mode", SEARCH_MODES, index=0)
                top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
                rerank = st.checkbox("Use CrossEncoder reranking", value=False)
            filters = render_filter_controls(
                documents,
                key_prefix="search",
                scope_mode=scope_mode,
            )
            submitted = st.form_submit_button("Search", type="primary", icon=":material/search:")
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
    with st.container(key="ask-page"):
        with st.container(
            key="ask-page-header",
            horizontal=True,
            gap="medium",
            vertical_alignment="center",
        ):
            with st.container(key="ask-page-header-copy", width="stretch"):
                st.markdown('<div class="di-eyebrow">GROUNDED DOCUMENT Q&A</div>', unsafe_allow_html=True)
                st.markdown('<h1 class="di-page-title">Ask AI</h1>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="di-page-subtitle">Ask questions, search document evidence and continue '
                    'grounded conversations with source-backed answers.</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="di-ask-header-icon">{INLINE_ICONS["chat"]}</div>',
                unsafe_allow_html=True,
            )

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
    st.markdown(
        f'<div class="di-ask-section-heading"><span class="di-ask-section-icon">'
        f'{INLINE_ICONS["chat"]}</span><div><h2>Grounded Q&A</h2>'
        '<p>Answers are generated only from evidence returned by the backend retrieval pipeline.</p></div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="ask-grounded-card", border=True):
        documents = _load_documents_for_filters(client)
        scope_mode = render_document_scope_selector(key_prefix="ask")
        with st.form("ask_form", border=False):
            st.markdown('<div class="di-form-section-label">Question</div>', unsafe_allow_html=True)
            question = st.text_area(
                "Question",
                placeholder="What does the indexed policy say?",
                height=140,
            )
            st.markdown('<div class="di-form-section-label di-form-section-label--settings">Retrieval settings</div>', unsafe_allow_html=True)
            with st.container(key="ask-retrieval-settings", horizontal=True, gap="medium"):
                mode = st.selectbox("Search mode", SEARCH_MODES, index=0, key="ask-mode")
                top_k = st.number_input("Evidence top K", min_value=1, max_value=50, value=5, key="ask-top-k")
                rerank = st.checkbox("Use CrossEncoder reranking", value=True, key="ask-rerank")
            filters = render_filter_controls(
                documents,
                key_prefix="ask",
                scope_mode=scope_mode,
            )
            submitted = st.form_submit_button("Ask", type="primary", icon=":material/auto_awesome:")
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
        with st.container(key="ask-answer-card", border=True):
            st.markdown(
                f'<div class="di-answer-heading"><span class="di-answer-icon">'
                f'{INLINE_ICONS["sparkle"]}</span><div><div class="di-answer-kicker">ANSWER</div>'
                '<h2>Grounded answer</h2></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(response.get("answer") or "No answer was returned.")
            if response.get("citations_valid") is False:
                st.warning("The backend flagged one or more unsupported source citations.")
        render_sources(response.get("sources"))
        render_timings(response, ("retrieval_time_ms", "rerank_time_ms", "generation_time_ms", "total_time_ms"))


def render_conversations(client: ApiClient, *, nested: bool = False) -> None:
    heading = "Conversations"
    copy = "Continue persistent document-grounded conversations with retained context."
    st.markdown(
        f'<div class="di-ask-section-heading"><span class="di-ask-section-icon">'
        f'{INLINE_ICONS["conversation"]}</span><div><h2>{heading}</h2><p>{copy}</p></div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="ask-conversations-card", border=True):
        if st.button("Refresh conversations", icon=":material/refresh:") or st.session_state.conversations is None:
            st.session_state.conversations = conversations_api.list_conversations(client)
        with st.form("create-conversation-form", clear_on_submit=True, border=False):
            st.markdown('<div class="di-form-section-label">Start a new conversation</div>', unsafe_allow_html=True)
            title = st.text_input("Optional conversation title", placeholder="e.g. Benefits policy review")
            create_clicked = st.form_submit_button("New conversation", type="primary", icon=":material/add_comment:")
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

        with st.container(key="conversation-history", border=True):
            st.markdown('<div class="di-form-section-label">Conversation history</div>', unsafe_allow_html=True)
            for message in st.session_state.conversation_messages:
                role = message.get("role", "assistant")
                role_label = "USER" if role == "user" else "DOCUINTEL"
                with st.chat_message(role):
                    st.markdown(f'<div class="di-chat-role">{role_label}</div>', unsafe_allow_html=True)
                    st.markdown(message.get("content") or "")

        with st.container(key="conversation-controls", border=True):
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

        with st.container(key="conversation-danger-zone", border=True):
            st.markdown('<div class="di-danger-heading">Delete conversation</div>', unsafe_allow_html=True)
            st.caption("This permanently removes the selected conversation and its retained history.")
            with st.form("delete-conversation-form", border=False):
                confirmed = st.checkbox("I understand this deletes the conversation")
                delete_clicked = st.form_submit_button("Delete conversation", icon=":material/delete:")
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
    with st.container(key=f"{key_prefix}-filter-panel", border=True):
        st.markdown(
            '<div class="di-filter-heading"><span class="di-filter-icon">'
            f'{INLINE_ICONS["filter"]}</span><div><strong>Advanced filters</strong>'
            '<span>Refine the evidence returned to the workspace.</span></div></div>',
            unsafe_allow_html=True,
        )
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
