"""Structure-aware chunk generation for persisted document extraction results."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A deterministic chunk ready for embedding and database persistence."""

    text: str
    start_page: int
    end_page: int
    page_id: UUID | None
    section_heading: str | None
    content_type: str
    contains_ocr: bool
    character_count: int
    token_count: int
    fingerprint_sha256: str


@dataclass(slots=True)
class _TextUnit:
    """One ordered source unit used while assembling a section."""

    text: str
    page_number: int
    page_id: UUID | None
    section_heading: str | None
    content_type: str
    contains_ocr: bool


class StructureAwareChunker:
    """Group persisted layout content by heading and split only at safe boundaries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        target_chars: int | None = None,
        max_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        configured = settings or Settings()
        self.target_chars = target_chars or configured.chunk_target_chars
        self.max_chars = max_chars or configured.chunk_max_chars
        self.overlap_chars = (
            configured.chunk_overlap_chars if overlap_chars is None else overlap_chars
        )
        if self.max_chars < self.target_chars:
            raise ValueError("chunk_max_chars must be greater than or equal to chunk_target_chars")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")

    def build_chunks(self, document: Any) -> list[ChunkDraft]:
        """Build ordered chunks from a loaded ORM document and its page structure."""

        drafts: list[ChunkDraft] = []
        current_heading: str | None = None
        pending: list[_TextUnit] = []

        def flush_pending() -> None:
            nonlocal pending
            if pending:
                drafts.extend(self._flush_text_units(pending))
                pending = []

        pages = sorted(getattr(document, "pages", []) or [], key=lambda page: page.page_number)
        for page in pages:
            page_id = getattr(page, "id", None)
            contains_ocr = bool(
                getattr(page, "ocr_applied", False)
                or getattr(page, "extraction_method", "") == "ocr"
            )
            elements = sorted(
                getattr(page, "layout_elements", []) or [],
                key=lambda element: getattr(element, "sequence_order", 0),
            )
            text_elements = [
                element
                for element in elements
                if self._normalise(getattr(element, "text", ""))
                and getattr(element, "element_type", "") != "table"
            ]
            if text_elements:
                for element in text_elements:
                    element_type = str(getattr(element, "element_type", "text"))
                    text = self._normalise(getattr(element, "text", ""))
                    if element_type == "heading":
                        flush_pending()
                        current_heading = text
                        continue
                    content_type = "list" if element_type == "list_item" else "text"
                    pending.extend(
                        self._units_for_text(
                            text=text,
                            page_number=page.page_number,
                            page_id=page_id,
                            section_heading=current_heading,
                            content_type=content_type,
                            contains_ocr=contains_ocr,
                        )
                    )
            elif not elements or not (getattr(page, "tables", []) or []):
                fallback_text = self._normalise(getattr(page, "extracted_text", ""))
                if fallback_text:
                    pending.extend(
                        self._units_for_text(
                            text=fallback_text,
                            page_number=page.page_number,
                            page_id=page_id,
                            section_heading=current_heading,
                            content_type="text",
                            contains_ocr=contains_ocr,
                        )
                    )

            # Tables are persisted separately from layout elements. They are
            # intentionally emitted as their own units so row/column structure
            # is retained and table text is not duplicated as paragraph text.
            for table in getattr(page, "tables", []) or []:
                table_text = self._table_text(table)
                if not table_text:
                    continue
                flush_pending()
                drafts.extend(
                    self._flush_text_units(
                        self._units_for_text(
                            text=table_text,
                            page_number=page.page_number,
                            page_id=page_id,
                            section_heading=current_heading,
                            content_type="table",
                            contains_ocr=contains_ocr,
                        )
                    )
                )

        flush_pending()
        logger.debug("Built %s structure-aware chunks document_id=%s", len(drafts), getattr(document, "id", None))
        return drafts

    def _flush_text_units(self, units: list[_TextUnit]) -> list[ChunkDraft]:
        """Assemble a same-section unit list into target-sized chunks."""

        if not units:
            return []
        drafts: list[ChunkDraft] = []
        buffer_text = ""
        buffer_units: list[_TextUnit] = []
        buffer_is_overlap = False

        def flush_buffer() -> None:
            nonlocal buffer_text, buffer_units, buffer_is_overlap
            if not buffer_text or not buffer_units:
                buffer_text = ""
                buffer_units = []
                buffer_is_overlap = False
                return
            drafts.append(self._draft_from_units(buffer_text, buffer_units))
            overlap = self._overlap_tail(buffer_text)
            if overlap:
                last = buffer_units[-1]
                buffer_text = overlap
                buffer_units = [
                    _TextUnit(
                        text=overlap,
                        page_number=last.page_number,
                        page_id=last.page_id,
                        section_heading=last.section_heading,
                        content_type=last.content_type,
                        contains_ocr=last.contains_ocr,
                    )
                ]
                buffer_is_overlap = True
            else:
                buffer_text = ""
                buffer_units = []
                buffer_is_overlap = False

        for unit in units:
            if not buffer_text:
                buffer_text = unit.text
                buffer_units = [unit]
                buffer_is_overlap = False
                continue
            candidate_length = len(buffer_text) + 1 + len(unit.text)
            if candidate_length > self.max_chars or (
                len(buffer_text) >= self.target_chars and candidate_length > self.target_chars
            ):
                flush_buffer()
                if len(buffer_text) + 1 + len(unit.text) > self.max_chars:
                    # The overlap itself can consume the available space for a
                    # short next unit. Start a clean chunk rather than exceeding
                    # the configured maximum.
                    buffer_text = ""
                    buffer_units = []
                    buffer_is_overlap = False
                if not buffer_text:
                    buffer_text = unit.text
                    buffer_units = [unit]
                    buffer_is_overlap = False
                else:
                    buffer_text = f"{buffer_text} {unit.text}"
                    buffer_units.append(unit)
                    buffer_is_overlap = False
            else:
                buffer_text = f"{buffer_text} {unit.text}"
                buffer_units.append(unit)
                buffer_is_overlap = False
        if buffer_text:
            # A trailing overlap is not useful as a standalone chunk.
            if not buffer_is_overlap:
                drafts.append(self._draft_from_units(buffer_text, buffer_units))
        return drafts

    def _units_for_text(
        self,
        *,
        text: str,
        page_number: int,
        page_id: UUID | None,
        section_heading: str | None,
        content_type: str,
        contains_ocr: bool,
    ) -> list[_TextUnit]:
        """Split a single oversized paragraph/list/table without losing its provenance."""

        if len(text) <= self.max_chars:
            return [
                _TextUnit(
                    text,
                    page_number,
                    page_id,
                    section_heading,
                    content_type,
                    contains_ocr,
                )
            ]
        pieces: list[_TextUnit] = []
        remaining = text
        while remaining:
            if len(remaining) <= self.max_chars:
                piece = remaining
                remaining = ""
            else:
                cut = self._preferred_cut(remaining[: self.max_chars + 1])
                piece = remaining[:cut].rstrip()
                remaining = remaining[cut:].lstrip()
                if not piece:
                    piece = remaining[: self.max_chars]
                    remaining = remaining[self.max_chars :].lstrip()
            pieces.append(
                _TextUnit(
                    piece,
                    page_number,
                    page_id,
                    section_heading,
                    content_type,
                    contains_ocr,
                )
            )
        return pieces

    def _draft_from_units(self, text: str, units: list[_TextUnit]) -> ChunkDraft:
        """Create a draft with stable provenance and content fingerprint."""

        normalised = self._normalise(text)
        start_page = min(unit.page_number for unit in units)
        end_page = max(unit.page_number for unit in units)
        content_types = {unit.content_type for unit in units}
        if content_types == {"table"}:
            content_type = "table"
        elif content_types == {"list"}:
            content_type = "list"
        elif len(content_types) == 1:
            content_type = "text"
        else:
            content_type = "mixed"
        heading = next((unit.section_heading for unit in units if unit.section_heading), None)
        fingerprint_source = "\n".join(
            [normalised, str(start_page), str(end_page), heading or "", content_type]
        )
        return ChunkDraft(
            text=normalised,
            start_page=start_page,
            end_page=end_page,
            page_id=units[0].page_id,
            section_heading=heading,
            content_type=content_type,
            contains_ocr=any(unit.contains_ocr for unit in units),
            character_count=len(normalised),
            token_count=len(normalised.split()),
            fingerprint_sha256=hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest(),
        )

    def _overlap_tail(self, text: str) -> str:
        """Return a word-aligned overlap tail for the next chunk."""

        if self.overlap_chars <= 0 or len(text) <= self.overlap_chars:
            return ""
        tail = text[-self.overlap_chars :]
        first_space = tail.find(" ")
        return tail[first_space + 1 :] if first_space >= 0 else tail

    @staticmethod
    def _preferred_cut(text: str) -> int:
        """Prefer sentence or word boundaries while guaranteeing progress."""

        sentence_matches = list(re.finditer(r"[.!?](?:\s|$)", text))
        if sentence_matches:
            return max(1, sentence_matches[-1].end() - 1)
        whitespace = text.rfind(" ")
        return max(1, whitespace)

    @staticmethod
    def _table_text(table: Any) -> str:
        """Render headers and rows as a stable row-oriented text representation."""

        headers = [StructureAwareChunker._normalise(str(value)) for value in (getattr(table, "headers", []) or [])]
        rows = [
            [StructureAwareChunker._normalise(str(value)) for value in row]
            for row in (getattr(table, "rows", []) or [])
        ]
        lines = [" | ".join(value for value in headers if value)] if headers else []
        lines.extend(" | ".join(value for value in row if value) for row in rows)
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _normalise(value: str) -> str:
        """Collapse extraction whitespace and discard empty content."""

        return " ".join(value.split())
