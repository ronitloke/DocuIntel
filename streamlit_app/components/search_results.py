"""Search result display components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_search_results(response: dict[str, Any] | None) -> None:
    if not response:
        return
    results = response.get("results") or []
    st.subheader(f"Results ({len(results)})")
    if not results:
        st.info("No indexed chunks matched this query.")
        return
    for result in results:
        rank = result.get("rank", "—")
        filename = result.get("original_filename") or "Unknown document"
        start = result.get("start_page")
        end = result.get("end_page")
        page = f"page {start}" if start == end or end is None else f"pages {start}–{end}"
        with st.expander(f"#{rank} · {filename} · {page}"):
            st.write(result.get("text") or "")
            cols = st.columns(4)
            cols[0].caption(f"Method: {result.get('retrieval_method', '—')}")
            cols[1].caption(f"Base rank: {result.get('base_rank', '—')}")
            cols[2].caption(f"Content: {result.get('content_type', '—')}")
            cols[3].caption(f"Rerank: {result.get('rerank_score', '—')}")

