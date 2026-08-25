"""Deterministic text and structured-table comparison primitives."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable
from uuid import UUID

from app.db.repository import PersistedTableRecord


@dataclass(frozen=True, slots=True)
class ComparisonBlock:
    """Detached chunk evidence used by the bounded alignment algorithm."""

    source_id: str
    document_id: UUID
    chunk_id: UUID
    filename: str
    sequence_number: int
    text: str
    normalized_text: str
    start_page: int | None
    end_page: int | None
    section_heading: str | None


@dataclass(frozen=True, slots=True)
class TextMatch:
    """One exact or conservative probable match between two chunks."""

    base: ComparisonBlock | None
    target: ComparisonBlock | None
    similarity: float | None


@dataclass(frozen=True, slots=True)
class TableMatch:
    """One conservatively matched table pair."""

    base: PersistedTableRecord | None
    target: PersistedTableRecord | None
    similarity: float | None


@dataclass(frozen=True, slots=True)
class ComparisonDiff:
    """All deterministic matches, including unchanged items hidden from the API by default."""

    text_matches: tuple[TextMatch, ...]
    table_matches: tuple[TableMatch, ...]


_TOKEN_PATTERN = re.compile(r"[\w]+(?:[-'][\w]+)?", re.UNICODE)


def normalize_text(value: str) -> str:
    """Remove extraction presentation noise while preserving meaningful wording and values."""

    normalized = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", normalized).strip()


def text_similarity(left: str, right: str) -> float:
    """Return a standard-library sequence similarity, explicitly labelled as similarity."""

    return round(SequenceMatcher(None, left, right).ratio(), 6)


def _tokens(value: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(value)}


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def align_text_blocks(
    base_blocks: Iterable[ComparisonBlock],
    target_blocks: Iterable[ComparisonBlock],
    *,
    similarity_threshold: float = 0.55,
) -> tuple[TextMatch, ...]:
    """Align exact content first, then pair only conservatively similar unmatched chunks.

    Page numbers are deliberately not used as the primary key. Exact normalized text can
    therefore be recognised after a page move, while modified pairing requires both a
    meaningful sequence similarity and shared content tokens.
    """

    base = sorted(base_blocks, key=lambda item: (item.sequence_number, item.source_id))
    target = sorted(target_blocks, key=lambda item: (item.sequence_number, item.source_id))
    unmatched_base = set(range(len(base)))
    unmatched_target = set(range(len(target)))
    matches: list[TextMatch] = []

    target_by_text: dict[str, list[int]] = defaultdict(list)
    for index, block in enumerate(target):
        target_by_text[block.normalized_text].append(index)

    # Queue order makes repeated/duplicate content deterministic.
    for base_index, block in enumerate(base):
        candidates = [index for index in target_by_text[block.normalized_text] if index in unmatched_target]
        if not candidates:
            continue
        target_index = min(candidates, key=lambda index: (target[index].sequence_number, target[index].source_id))
        unmatched_base.remove(base_index)
        unmatched_target.remove(target_index)
        matches.append(TextMatch(base=block, target=target[target_index], similarity=None))

    # Greedy highest-score pairing is bounded, explainable, and avoids an ML dependency.
    for base_index in sorted(unmatched_base, key=lambda index: (base[index].sequence_number, base[index].source_id)):
        candidates: list[tuple[float, float, int]] = []
        for target_index in unmatched_target:
            similarity = text_similarity(base[base_index].normalized_text, target[target_index].normalized_text)
            overlap = _token_overlap(base[base_index].normalized_text, target[target_index].normalized_text)
            if similarity >= similarity_threshold and overlap >= 0.6:
                candidates.append((similarity, overlap, target_index))
        if not candidates:
            continue
        _, _, target_index = max(
            candidates,
            key=lambda item: (item[0], item[1], -target[item[2]].sequence_number, target[item[2]].source_id),
        )
        unmatched_base.remove(base_index)
        unmatched_target.remove(target_index)
        matches.append(
            TextMatch(
                base=base[base_index],
                target=target[target_index],
                similarity=text_similarity(base[base_index].normalized_text, target[target_index].normalized_text),
            )
        )

    matches.extend(TextMatch(base=base[index], target=None, similarity=None) for index in unmatched_base)
    matches.extend(TextMatch(base=None, target=target[index], similarity=None) for index in unmatched_target)
    return tuple(
        sorted(
            matches,
            key=lambda item: (
                item.base.sequence_number if item.base is not None else 10**9,
                item.target.sequence_number if item.target is not None else 10**9,
                (item.base.source_id if item.base is not None else item.target.source_id),
            ),
        )
    )


def _normalized_headers(table: PersistedTableRecord) -> list[str]:
    return [normalize_text(header).casefold() for header in table.headers]


def _row_keys(table: PersistedTableRecord) -> set[str]:
    if not table.headers:
        return set()
    return {
        normalize_text(row[0]).casefold()
        for row in table.rows
        if row and normalize_text(row[0])
    }


def _table_similarity(base: PersistedTableRecord, target: PersistedTableRecord) -> float:
    base_headers = set(_normalized_headers(base))
    target_headers = set(_normalized_headers(target))
    if not base_headers or not target_headers:
        return 0.0
    header_overlap = len(base_headers & target_headers) / max(len(base_headers), len(target_headers))
    base_keys = _row_keys(base)
    target_keys = _row_keys(target)
    row_overlap = (
        len(base_keys & target_keys) / max(len(base_keys), len(target_keys))
        if base_keys and target_keys
        else 0.0
    )
    page_proximity = max(0.0, 1.0 - abs(base.page_number - target.page_number) / 5)
    order_match = 1.0 if base.table_index == target.table_index else 0.0
    return round(0.55 * header_overlap + 0.3 * row_overlap + 0.1 * page_proximity + 0.05 * order_match, 6)


def align_tables(
    base_tables: Iterable[PersistedTableRecord],
    target_tables: Iterable[PersistedTableRecord],
) -> tuple[TableMatch, ...]:
    """Match tables only when headers and identity/order signals support the relationship."""

    base = sorted(base_tables, key=lambda item: (item.page_number, item.table_index, str(item.table_id)))
    target = sorted(target_tables, key=lambda item: (item.page_number, item.table_index, str(item.table_id)))
    unmatched_base = set(range(len(base)))
    unmatched_target = set(range(len(target)))
    matches: list[TableMatch] = []
    for base_index in range(len(base)):
        candidates = [
            ( _table_similarity(base[base_index], target[target_index]), target_index)
            for target_index in unmatched_target
        ]
        candidates = [item for item in candidates if item[0] >= 0.60]
        if not candidates:
            continue
        score, target_index = max(
            candidates,
            key=lambda item: (item[0], -target[item[1]].page_number, -target[item[1]].table_index, str(target[item[1]].table_id)),
        )
        unmatched_base.remove(base_index)
        unmatched_target.remove(target_index)
        matches.append(TableMatch(base=base[base_index], target=target[target_index], similarity=score))

    matches.extend(TableMatch(base=base[index], target=None, similarity=None) for index in unmatched_base)
    matches.extend(TableMatch(base=None, target=target[index], similarity=None) for index in unmatched_target)
    return tuple(
        sorted(
            matches,
            key=lambda item: (
                item.base.page_number if item.base is not None else 10**9,
                item.target.page_number if item.target is not None else 10**9,
                (str(item.base.table_id) if item.base is not None else str(item.target.table_id)),
            ),
        )
    )


def compare_content(
    base_blocks: Iterable[ComparisonBlock],
    target_blocks: Iterable[ComparisonBlock],
    base_tables: Iterable[PersistedTableRecord],
    target_tables: Iterable[PersistedTableRecord],
) -> ComparisonDiff:
    """Run the deterministic text and table alignment stages."""

    return ComparisonDiff(
        text_matches=align_text_blocks(base_blocks, target_blocks),
        table_matches=align_tables(base_tables, target_tables),
    )
