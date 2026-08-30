"""Evidence/source display components."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


EVIDENCE_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M6 3.5h8l4 4V20.5H6z"/><path d="M14 3.5v4h4"/>'
    '<path d="M9 12h6M9 15.5h6"/></svg>'
)


def render_sources(sources: list[dict[str, Any]] | None) -> None:
    """Render compact source evidence without exposing full documents."""

    if not sources:
        st.info("No source metadata was returned.")
        return
    st.markdown(
        '<div class="di-evidence-section-heading"><span class="di-evidence-section-icon">'
        f'{EVIDENCE_ICON}</span>'
        '<div><h2>Sources</h2><p>Evidence returned by the grounded retrieval pipeline.</p></div></div>',
        unsafe_allow_html=True,
    )
    for index, source in enumerate(sources, start=1):
        source_id = str(source.get("source_id") or f"S{index}")
        filename = str(source.get("filename") or "Unknown document")
        page_text = _page_label(source)
        with st.expander(f"{source_id} · {filename} · {page_text}"):
            st.markdown(
                f'<div class="di-evidence-card-heading"><span class="di-source-badge">'
                f'{escape(source_id)}</span><div><strong>{escape(filename)}</strong>'
                f'<span>{escape(page_text)}</span></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="di-evidence-label">Evidence</div>', unsafe_allow_html=True)
            st.write(source.get("excerpt") or "No excerpt returned.")
            score = source.get("reranker_score")
            st.markdown(
                '<div class="di-evidence-meta">'
                f'<span>Final rank: <strong>{escape(str(source.get("final_rank", "—")))}</strong></span>'
                f'<span>Chunk: <strong>{escape(str(source.get("chunk_id", "—")))}</strong></span>'
                f'<span>Reranker score: <strong>{escape(str(score if score is not None else "—"))}</strong></span>'
                '</div>',
                unsafe_allow_html=True,
            )


def _page_label(source: dict[str, Any]) -> str:
    start = source.get("start_page")
    end = source.get("end_page")
    if start is None:
        return "page unavailable"
    return f"page {start}" if start == end or end is None else f"pages {start}–{end}"
