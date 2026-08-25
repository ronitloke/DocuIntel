"""Fail-closed loading and validation for authoritative E5 inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.e5.models import BaselineArtifact


class E5ArtifactError(ValueError):
    """Raised when the explicit E5 baseline cannot be trusted."""


_EXPECTED = {
    "e2_funsd": ("E2", "funsd", "test"),
    "e2_doclaynet": ("E2", "doclaynet", "validation"),
    "e3": ("E3", "docvqa", "validation"),
    "e4": ("E4", "docvqa", "validation"),
    "e4_1": ("E4.1", "docvqa", "validation"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise E5ArtifactError(f"Referenced artifact does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E5ArtifactError(f"Could not parse JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E5ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _resolve_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _infer_run_id(module: str, summary_path: Path) -> str:
    if module == "E2":
        return summary_path.parent.parent.name
    return summary_path.parent.name


def _validate_source(key: str, source: dict[str, Any], *, root: Path) -> BaselineArtifact:
    if key not in _EXPECTED:
        raise E5ArtifactError(f"Unexpected baseline source key: {key}")
    expected_module, expected_dataset, expected_split = _EXPECTED[key]
    required = {"module", "dataset", "split", "run_id", "summary", "run_metadata"}
    missing = sorted(required - source.keys())
    if missing:
        raise E5ArtifactError(f"Baseline source {key} is missing fields: {', '.join(missing)}")
    if (source["module"], source["dataset"], source["split"]) != (
        expected_module,
        expected_dataset,
        expected_split,
    ):
        raise E5ArtifactError(f"Baseline source {key} has inconsistent module/dataset/split metadata")

    summary_path = _resolve_path(str(source["summary"]), root=root)
    metadata_path = _resolve_path(str(source["run_metadata"]), root=root)
    summary = _read_json(summary_path)
    metadata = _read_json(metadata_path)
    if summary.get("dataset") != expected_dataset or summary.get("split") != expected_split:
        raise E5ArtifactError(f"Summary metadata does not match the E5 manifest for {key}")
    if metadata.get("dataset") not in (None, expected_dataset) or metadata.get("split") not in (
        None,
        expected_split,
    ):
        raise E5ArtifactError(f"Run metadata does not match the E5 manifest for {key}")
    schema_prefix = {"E2": "e2.", "E3": "e3.", "E4": "e4.", "E4.1": "e4_1."}[expected_module]
    schema = str(summary.get("schema_version", ""))
    if not schema.startswith(schema_prefix):
        raise E5ArtifactError(f"Unexpected summary schema for {key}: {schema!r}")
    metadata_schema = str(metadata.get("schema_version", ""))
    if metadata_schema and not metadata_schema.startswith(schema_prefix):
        raise E5ArtifactError(f"Unexpected metadata schema for {key}: {metadata_schema!r}")

    run_id = str(source["run_id"])
    if _infer_run_id(expected_module, summary_path) != run_id:
        raise E5ArtifactError(
            f"Manifest run_id {run_id!r} does not match the explicitly referenced path {summary_path}"
        )
    if metadata.get("run_id") not in (None, run_id):
        raise E5ArtifactError(f"Run metadata run_id does not match the E5 manifest for {key}")
    if expected_module == "E3" and summary.get("status") != "completed":
        raise E5ArtifactError("The selected E3 artifact is not a completed real run")
    if expected_module == "E4" and summary.get("status") != "completed":
        raise E5ArtifactError("The selected E4 artifact is not a completed production baseline")
    if expected_module == "E4.1" and summary.get("status") not in {"CONTROLLED_FAILURE", "BLOCKED"}:
        raise E5ArtifactError("The selected E4.1 artifact is not the controlled blocked diagnostic")
    supplemental: dict[str, Path] = {}
    for name, value in source.get("supplemental", {}).items():
        path = _resolve_path(str(value), root=root)
        if not path.is_file():
            raise E5ArtifactError(f"Referenced supplemental artifact does not exist: {path}")
        supplemental[name] = path
    return BaselineArtifact(
        key=key,
        module=expected_module,
        dataset=expected_dataset,
        split=expected_split,
        run_id=run_id,
        summary_path=summary_path,
        metadata_path=metadata_path,
        summary=summary,
        metadata=metadata,
        supplemental_paths=supplemental,
    )


def load_baseline(manifest_path: Path, *, project_root: Path | None = None) -> dict[str, BaselineArtifact]:
    """Load and validate every explicit source in an E5 manifest."""

    manifest_path = manifest_path.resolve()
    root = (project_root or manifest_path.parents[2]).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "e5.baseline.v1":
        raise E5ArtifactError("Unsupported or missing E5 baseline manifest schema")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise E5ArtifactError("E5 baseline manifest must contain a sources object")
    if set(sources) != set(_EXPECTED):
        missing = sorted(set(_EXPECTED) - set(sources))
        extra = sorted(set(sources) - set(_EXPECTED))
        raise E5ArtifactError(f"E5 baseline source keys differ; missing={missing}, extra={extra}")
    return {key: _validate_source(key, value, root=root) for key, value in sources.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load an optional JSONL artifact while rejecting malformed records."""

    if not path.is_file():
        raise E5ArtifactError(f"Referenced JSONL artifact does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise E5ArtifactError(f"Malformed JSONL record {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise E5ArtifactError(f"JSONL record must be an object {path}:{line_number}")
        records.append(value)
    return records


def require_path(value: Any, path: str, *, source: str) -> Any:
    """Read a required nested artifact value and fail instead of returning a fake N/A."""

    current = value
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            raise E5ArtifactError(f"Required metric {path!r} is missing from {source}")
        current = current[segment]
    return current
