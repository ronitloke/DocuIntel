"""Evaluation E5 final benchmark consolidation."""

from evaluation.e5.builder import build_e5
from evaluation.e5.loader import E5ArtifactError, load_baseline

__all__ = ["E5ArtifactError", "build_e5", "load_baseline"]
