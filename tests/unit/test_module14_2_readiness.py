"""Focused Module 14.2 readiness and deployment-diagnostic tests."""

from __future__ import annotations

from app.db.health import check_database
from scripts import check_deployment
from scripts.check_deployment import HttpResult, run_checks
from streamlit_app.api.client import ApiError
from streamlit_app.ui_helpers import friendly_api_error_message


class _Connection:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def __enter__(self) -> "_Connection":
        if self.error:
            raise self.error
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, _statement: object) -> None:
        if self.error:
            raise self.error


class _Engine:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def connect(self) -> _Connection:
        return _Connection(self.error)


class _Database:
    def __init__(self, error: Exception | None = None) -> None:
        self.engine = _Engine(error)


def test_database_readiness_success() -> None:
    """A trivial query is sufficient to report database readiness."""

    assert check_database(_Database()) is True


def test_database_readiness_failure_logs_no_connection_details(caplog) -> None:
    """Database failures become false without putting exception text in logs."""

    secret_error = RuntimeError("postgresql://user:secret-password@example.invalid/db")
    with caplog.at_level("WARNING"):
        assert check_database(_Database(secret_error)) is False

    assert "PostgreSQL readiness check failed" in caplog.text
    assert "secret-password" not in caplog.text
    assert "example.invalid" not in caplog.text


def _successful_fetch(url: str, _timeout: float) -> HttpResult:
    if url.endswith("/health"):
        return HttpResult(200, {"status": "healthy"})
    if url.endswith("/ready"):
        return HttpResult(200, {"status": "healthy", "database": "healthy"})
    if url.endswith("/_stcore/health"):
        return HttpResult(200, "ok")
    return HttpResult(200, {"models": [{"name": "llama3.2:3b"}]})


def test_diagnostic_success_and_optional_ollama_check() -> None:
    """All required checks pass and the optional model is reported."""

    checks = run_checks(
        api_url="http://api",
        streamlit_url="http://frontend",
        ollama_url="http://ollama",
        ollama_model="llama3.2:3b",
        fetcher=_successful_fetch,
    )

    assert [(check.label, check.status) for check in checks] == [
        ("FastAPI health", "OK"),
        ("FastAPI readiness (PostgreSQL)", "OK"),
        ("Streamlit", "OK"),
        ("Ollama", "OK"),
    ]


def test_diagnostic_required_failure_returns_nonzero_without_secrets(monkeypatch, capsys) -> None:
    """Required readiness failure fails diagnostics while optional Ollama does not."""

    def failing_fetch(url: str, _timeout: float) -> HttpResult:
        if url.endswith("/health"):
            return HttpResult(200, {"status": "healthy"})
        if url.endswith("/ready"):
            return HttpResult(503, {"status": "unready", "database": "unavailable"})
        if url.endswith("/_stcore/health"):
            return HttpResult(200, "ok")
        return HttpResult(None, error_kind="connection_failure")

    checks = run_checks(
        api_url="http://api",
        streamlit_url="http://frontend",
        ollama_url="http://ollama",
        ollama_model="llama3.2:3b",
        fetcher=failing_fetch,
    )
    assert checks[1].detail == "database unavailable"
    assert checks[3].status == "WARN"

    monkeypatch.setattr(check_deployment, "fetch_url", failing_fetch)
    exit_code = check_deployment.main(
        ["--api-url", "http://api", "--streamlit-url", "http://frontend", "--ollama-url", "http://ollama"]
    )
    assert exit_code == 1

    assert friendly_api_error_message(
        ApiError("The local Ollama service is unavailable.", status_code=503)
    ) == "The local language model is unavailable. Start Ollama and try again."

    output = capsys.readouterr().out
    assert "secret" not in output


def test_diagnostic_optional_ollama_failure_keeps_core_exit_zero(monkeypatch, capsys) -> None:
    """An Ollama outage is a warning when required services remain healthy."""

    def optional_failure_fetch(url: str, _timeout: float) -> HttpResult:
        if url.endswith("/health"):
            return HttpResult(200, {"status": "healthy"})
        if url.endswith("/ready"):
            return HttpResult(200, {"status": "healthy", "database": "healthy"})
        if url.endswith("/_stcore/health"):
            return HttpResult(200, "ok")
        return HttpResult(None, error_kind="connection_failure")

    monkeypatch.setattr(check_deployment, "fetch_url", optional_failure_fetch)
    assert check_deployment.main(
        ["--api-url", "http://api", "--streamlit-url", "http://frontend", "--ollama-url", "http://ollama"]
    ) == 0
    assert "optional features may be unavailable" in capsys.readouterr().out


def test_diagnostic_check_result_payloads_are_mapping_safe() -> None:
    """Diagnostic parsing never assumes provider payloads have a dictionary shape."""

    def malformed_fetch(url: str, _timeout: float) -> HttpResult:
        if url.endswith("/health") or url.endswith("/ready"):
            return HttpResult(200, [])
        if url.endswith("/_stcore/health"):
            return HttpResult(200, "ok")
        return HttpResult(200, [])

    checks = run_checks(
        api_url="http://api",
        streamlit_url="http://frontend",
        ollama_url="http://ollama",
        ollama_model="llama3.2:3b",
        fetcher=malformed_fetch,
    )
    assert checks[0].detail == "unexpected response"
    assert checks[1].detail == "unexpected response"
    assert checks[3].status == "WARN"
