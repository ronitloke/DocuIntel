"""Evidence/source display components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources(sources: list[dict[str, Any]] | None) -> None:
    """Render compact source evidence without exposing full documents."""

    if not sources:
        st.info("No source metadata was returned.")
        return
    st.subheader("Sources")
    for source in sources:
        source_id = source.get("source_id", "S?")
        filename = source.get("filename") or "Unknown document"
        page_text = _page_label(source)
        with st.expander(f"{source_id} · {filename} · {page_text}"):
            st.write(source.get("excerpt") or "No excerpt returned.")
            cols = st.columns(3)
            cols[0].caption(f"Final rank: {source.get('final_rank', '—')}")
            cols[1].caption(f"Chunk: {source.get('chunk_id', '—')}")
            score = source.get("reranker_score")
            cols[2].caption(f"Reranker score: {score if score is not None else '—'}")


def _page_label(source: dict[str, Any]) -> str:
    start = source.get("start_page")
    end = source.get("end_page")
    if start is None:
        return "page unavailable"
    return f"page {start}" if start == end or end is None else f"pages {start}–{end}"

