"""Typed records used by the read-only E5 consolidation package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


MeasurementStatus = Literal["MEASURED", "BLOCKED", "NOT_APPLICABLE", "NOT_MEASURED"]


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One provenance-preserving consolidated metric."""

    module: str
    dataset: str
    split: str
    run_id: str
    metric: str
    value: float | int | bool | None
    denominator: str | None
    status: MeasurementStatus
    source: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    """One explicitly selected authoritative source artifact."""

    key: str
    module: str
    dataset: str
    split: str
    run_id: str
    summary_path: Any
    metadata_path: Any
    summary: dict[str, Any]
    metadata: dict[str, Any]
    supplemental_paths: dict[str, Any]


@dataclass(frozen=True, slots=True)
class E5BuildResult:
    """Paths and summary returned by the E5 builder."""

    output_directory: Any
    summary: dict[str, Any]

