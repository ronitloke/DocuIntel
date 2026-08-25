"""Deterministic context construction for grounded RAG prompts."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.search import SearchResult


@dataclass(frozen=True, slots=True)
class ContextSource:
    """A final search result assigned a stable prompt source label."""

    source_id: str
    result: SearchResult
    excerpt: str


@dataclass(frozen=True, slots=True)
class RAGContext:
    """Bounded context text and the source metadata it contains."""

    text: str
    sources: list[ContextSource]


class RAGContextBuilder:
    """Build final-rank-ordered, character-bounded source blocks."""

    SOURCE_EXCERPT_CHARS = 500

    def build(self, results: list[SearchResult], *, max_chars: int) -> RAGContext:
        """Assign S labels and include complete blocks until the budget is reached."""

        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero.")

        blocks: list[str] = []
        sources: list[ContextSource] = []
        used_chars = 0
        for result in results:
            source_id = f"S{len(sources) + 1}"
            header = self._header(source_id, result)
            separator = "\n\n" if blocks else ""
            available = max_chars - used_chars - len(separator)
            full_block = f"{header}{result.text}"
            if len(full_block) <= available:
                block = full_block
                included_text = result.text
            elif not blocks:
                text_budget = max(0, available - len(header))
                if text_budget == 0:
                    break
                included_text = result.text[:text_budget]
                block = f"{header}{included_text}"
            else:
                break

            blocks.append(f"{separator}{block}")
            used_chars += len(separator) + len(block)
            sources.append(
                ContextSource(
                    source_id=source_id,
                    result=result,
                    excerpt=included_text[: self.SOURCE_EXCERPT_CHARS],
                )
            )

        return RAGContext(text="".join(blocks), sources=sources)

    @staticmethod
    def _header(source_id: str, result: SearchResult) -> str:
        """Format metadata without inventing unavailable page values."""

        page = RAGContextBuilder._page_text(result)
        section = result.section_heading or "(none)"
        score = (
            f"{result.rerank_score:.6f}"
            if result.rerank_score is not None
            else "(not reranked)"
        )
        return (
            f"[{source_id}]\n"
            f"Document: {result.original_filename}\n"
            f"Document ID: {result.document_id}\n"
            f"Chunk ID: {result.chunk_id}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Final rank: {result.rank}\n"
            f"Reranker score: {score}\n"
            "Content:\n"
        )

    @staticmethod
    def _page_text(result: SearchResult) -> str:
        """Describe only page metadata that exists on the search result."""

        if result.start_page is None:
            return "(unavailable)"
        if result.end_page is None or result.end_page == result.start_page:
            return str(result.start_page)
        return f"{result.start_page}-{result.end_page}"
