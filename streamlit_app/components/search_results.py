"""Search result display components."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def render_search_results(response: dict[str, Any] | None) -> None:
    if not response:
        return
    results = response.get("results") or []
    st.markdown(
        f'<div class="di-results-heading"><span class="di-results-count">{len(results)}</span>'
        f'<div><h2>Search results</h2><p>Retrieved evidence ranked by the selected search configuration.</p></div></div>',
        unsafe_allow_html=True,
    )
    if not results:
        st.info("No indexed chunks matched this query.")
        return
    for index, result in enumerate(results, start=1):
        rank = result.get("rank", "—")
        filename = result.get("original_filename") or "Unknown document"
        start = result.get("start_page")
        end = result.get("end_page")
        page = f"page {start}" if start == end or end is None else f"pages {start}–{end}"
        with st.expander(f"#{rank} · {filename} · {page}"):
            st.markdown(
                f'<div class="di-evidence-card-heading"><span class="di-source-badge">'
                f'R{escape(str(index))}</span><div><strong>{escape(str(filename))}</strong>'
                f'<span>{escape(page)}</span></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="di-evidence-label">Retrieved content</div>', unsafe_allow_html=True)
            st.write(result.get("text") or "")
            st.markdown(
                '<div class="di-evidence-meta">'
                f'<span>Method: <strong>{escape(str(result.get("retrieval_method", "—")))}</strong></span>'
                f'<span>Base rank: <strong>{escape(str(result.get("base_rank", "—")))}</strong></span>'
                f'<span>Content: <strong>{escape(str(result.get("content_type", "—")))}</strong></span>'
                f'<span>Rerank: <strong>{escape(str(result.get("rerank_score", "—")))}</strong></span>'
                '</div>',
                unsafe_allow_html=True,
            )
