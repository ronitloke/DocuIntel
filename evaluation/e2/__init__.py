"""Quantitative document-understanding evaluation for Module E2."""

from evaluation.e2.metrics import (
    LayoutEvaluation,
    TextEvaluation,
    evaluate_layout,
    evaluate_text,
    normalize_text,
)

__all__ = [
    "LayoutEvaluation",
    "TextEvaluation",
    "evaluate_layout",
    "evaluate_text",
    "normalize_text",
]
