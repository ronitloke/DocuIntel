"""High-value Module 12 contract tests for operational and presentation paths."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.evaluation.models import (
    ComparisonReport,
    EvaluationConfiguration,
    RetrievalEvaluationReport,
    RetrievalSummary,
)
from app.evaluation.reporting import attach_baseline, console_summary
from app.main import create_app
from streamlit_app.components import status as status_component


class _Connection:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def __enter__(self) -> "_Connection":
        if self.fail:
            raise ConnectionError("database unavailable")
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, _statement: object) -> None:
        if self.fail:
            raise ConnectionError("database unavailable")


class _Engine:
    def __init__(self) -> None:
        self.fail = False
        self.disposed = False

    def connect(self) -> _Connection:
        return _Connection(fail=self.fail)

    def dispose(self) -> None:
        self.disposed = True


def test_readiness_returns_503_without_exposing_database_exception(tmp_path) -> None:
    """Readiness failure is controlled and the process still shuts down cleanly."""

    engine = _Engine()
    database = SimpleNamespace(engine=engine)
    settings = Settings(app_name="Quality Test", database_url=None)
    application = create_app(
        settings=settings,
        storage_directory=tmp_path,
        database=database,
    )

    with TestClient(application) as client:
        healthy = client.get("/ready")
        assert healthy.status_code == 200
        assert healthy.json()["database"] == "healthy"

        engine.fail = True
        unavailable = client.get("/ready")

    assert unavailable.status_code == 503
    assert unavailable.json() == {
        "status": "unready",
        "service": "Quality Test",
        "version": "0.1.0",
        "database": "unavailable",
    }
    assert "database unavailable" not in unavailable.text
    assert engine.disposed is True


def _retrieval_report(mode: str, *, rerank: bool = False, mrr: float = 0.5) -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport(
        dataset="quality",
        configuration=EvaluationConfiguration(mode=mode, rerank=rerank, top_k=5),
        generated_at=datetime.now(UTC),
        summary=RetrievalSummary(
            cases=2,
            positive_cases=2,
            no_evidence_cases=0,
            success_at_k={"1": mrr, "3": 1.0},
            recall_at_k={"1": mrr, "3": 1.0},
            mrr=mrr,
            mean_retrieval_latency_ms=3.0,
            median_retrieval_latency_ms=3.0,
            mean_rerank_latency_ms=1.0 if rerank else None,
        ),
        cases=[],
    )


def test_reporting_covers_comparison_console_and_baseline_attachment(tmp_path) -> None:
    """Comparison output and baseline deltas remain machine- and human-readable."""

    semantic = _retrieval_report("semantic", mrr=0.5)
    hybrid = _retrieval_report("hybrid", rerank=True, mrr=0.75)
    comparison = ComparisonReport(
        dataset="quality",
        generated_at=datetime.now(UTC),
        reports=[semantic, hybrid],
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        '{"reports": [{"configuration": {"mode": "semantic", "rerank": false}, '
        '"summary": {"mrr": 0.4, "mean_total_search_latency_ms": 3.0}}]}\n',
        encoding="utf-8",
    )

    attach_baseline(comparison, baseline_path, tolerance=0.0)
    rendered = console_summary(comparison)

    assert "Dataset: quality" in rendered
    assert "semantic" in rendered
    assert "hybrid + rerank" in rendered
    assert comparison.baseline_comparison is not None
    assert "semantic" in comparison.baseline_comparison


def test_streamlit_status_adapter_handles_partial_backend_failure(monkeypatch) -> None:
    """The HTTP-only UI reports health and readiness independently."""

    class FakeClient:
        def get(self, path: str):
            if path == "/health":
                return {"status": "healthy", "version": "0.1.0"}
            raise RuntimeError("not ready")

    class FakeSidebar:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def success(self, message: str) -> None:
            self.messages.append(("success", message))

        def error(self, message: str) -> None:
            self.messages.append(("error", message))

        def warning(self, message: str) -> None:
            self.messages.append(("warning", message))

    # The adapter intentionally catches ApiError, matching the UI boundary.
    from streamlit_app.api.client import ApiError

    class ApiErrorClient(FakeClient):
        def get(self, path: str):
            if path == "/health":
                return super().get(path)
            raise ApiError("Database is not ready.")

    sidebar = FakeSidebar()
    monkeypatch.setattr(status_component.st, "sidebar", sidebar)
    snapshot = status_component.get_backend_status(ApiErrorClient())
    status_component.render_backend_status(snapshot)

    assert snapshot["health"]["status"] == "healthy"
    assert snapshot["ready_error"] == "Database is not ready."
    assert ("success", "API online · 0.1.0") in sidebar.messages
    assert any(
        level == "warning" and "Database is not ready." in message
        for level, message in sidebar.messages
    )


@pytest.mark.parametrize(
    ("value", "field"),
    [
        ("not-a-url", "ollama_base_url"),
        ("   ", "ollama_model"),
    ],
)
def test_settings_reject_invalid_provider_configuration(value: str, field: str) -> None:
    """Provider configuration fails early instead of failing during a request."""

    with pytest.raises(ValueError):
        Settings(**{field: value})
