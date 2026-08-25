"""Safe coordinate resolution for native PDF text detections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pymupdf as fitz


def resolve_word_coordinates(page: fitz.Page, matched_text: str) -> list[float] | None:
    """Resolve a detected native-text span to the union of exact PDF words.

    The resolver compares normalized consecutive PyMuPDF words. It never uses
    character counts or estimated rectangles, and returns ``None`` when the
    physical span cannot be proven from the source PDF.
    """

    try:
        raw_words = page.get_text("words") or []
    except (RuntimeError, ValueError):
        return None
    words = [
        _Word(
            text=str(item[4]),
            bbox=(float(item[0]), float(item[1]), float(item[2]), float(item[3])),
            block=int(item[5]) if len(item) > 5 else 0,
            line=int(item[6]) if len(item) > 6 else 0,
        )
        for item in raw_words
        if len(item) >= 5 and str(item[4]).strip()
    ]
    needle = _normalize(matched_text)
    if not needle or not words:
        return None

    for start in range(len(words)):
        joined: list[str] = []
        for end in range(start, min(len(words), start + 10)):
            joined.append(words[end].text)
            candidate = _normalize(" ".join(joined))
            if candidate == needle:
                return _union_bbox(words[start : end + 1])
            if len(candidate) > len(needle) + max(8, len(needle) // 3):
                break
    return None


class _Word:
    """Small internal representation of a PyMuPDF word tuple."""

    def __init__(self, text: str, bbox: tuple[float, float, float, float], block: int, line: int) -> None:
        self.text = text
        self.bbox = bbox
        self.block = block
        self.line = line


def _union_bbox(words: Sequence[_Word]) -> list[float] | None:
    if not words:
        return None
    return [
        min(word.bbox[0] for word in words),
        min(word.bbox[1] for word in words),
        max(word.bbox[2] for word in words),
        max(word.bbox[3] for word in words),
    ]


def _normalize(value: Any) -> str:
    return " ".join(str(value).split()).casefold()
