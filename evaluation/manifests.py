"""Deterministic JSONL manifest and preparation-metadata helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from evaluation.schemas import EvaluationDocument, PreparationMetadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def stable_evaluation_id(dataset: str, split: str, source_record_id: str) -> str:
    """Build a stable ID without random UUIDs or timestamps."""

    key = f"{dataset}\0{split}\0{source_record_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:20]
    return f"{dataset}-{digest}"


def repository_relative(path: Path) -> str:
    """Return a repository-relative POSIX path where possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_manifest(records: Iterable[EvaluationDocument], path: Path) -> Path:
    """Write normalized records in deterministic evaluation-ID order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.evaluation_id)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in ordered:
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            stream.write("\n")
    return path


def write_preparation_metadata(metadata: PreparationMetadata, path: Path) -> Path:
    """Write human-readable preparation provenance beside a manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def now_utc_iso() -> str:
    """Return an explicit UTC timestamp for provenance metadata only."""

    return datetime.now(UTC).isoformat()


def command_options(**values: Any) -> dict[str, Any]:
    """Drop absent optional CLI values while preserving explicit false/zero values."""

    return {key: value for key, value in values.items() if value is not None}

