"""Run lightweight, safe checks against a local DocuIntel deployment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://127.0.0.1:8001"
DEFAULT_STREAMLIT_URL = "http://127.0.0.1:8501"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
CHECK_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True, slots=True)
class HttpResult:
    """Safe result from one deployment HTTP probe."""

    status_code: int | None
    payload: Any = None
    error_kind: str | None = None


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One human-readable required or optional deployment check."""

    label: str
    status: str
    detail: str
    required: bool


Fetcher = Callable[[str, float], HttpResult]


def fetch_url(url: str, timeout_seconds: float = CHECK_TIMEOUT_SECONDS) -> HttpResult:
    """Fetch one endpoint without returning response bodies or secrets on failure."""

    request = Request(url, headers={"Accept": "application/json, text/plain"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read()
            payload: Any = None
            if raw_body:
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
            return HttpResult(status_code=response.getcode(), payload=payload)
    except HTTPError as exc:
        return HttpResult(
            status_code=exc.code,
            payload=_safe_error_payload(exc),
            error_kind="http_error",
        )
    except (TimeoutError, URLError, OSError, ValueError):
        return HttpResult(status_code=None, error_kind="connection_failure")


def _safe_error_payload(error: HTTPError) -> dict[str, str]:
    """Keep only known status fields from an HTTP error response."""

    try:
        decoded = json.loads(error.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(decoded, Mapping):
        return {}
    return {
        key: str(decoded[key])
        for key in ("status", "database")
        if key in decoded and isinstance(decoded[key], str)
    }


def run_checks(
    *,
    api_url: str,
    streamlit_url: str,
    ollama_url: str,
    ollama_model: str,
    fetcher: Fetcher | None = None,
) -> list[CheckResult]:
    """Run required API/UI checks and one optional Ollama check."""

    probe = fetcher or fetch_url
    results = [
        _check_status_endpoint(
            label="FastAPI health",
            url=f"{api_url.rstrip('/')}/health",
            expected_status="healthy",
            fetcher=probe,
        ),
        _check_readiness(api_url, probe),
        _check_streamlit(streamlit_url, probe),
        _check_ollama(ollama_url, ollama_model, probe),
    ]
    return results


def _check_status_endpoint(
    *,
    label: str,
    url: str,
    expected_status: str,
    fetcher: Fetcher,
) -> CheckResult:
    result = fetcher(url, CHECK_TIMEOUT_SECONDS)
    if result.status_code != 200:
        return CheckResult(label, "FAIL", _failure_detail(result), True)
    if not isinstance(result.payload, Mapping) or result.payload.get("status") != expected_status:
        return CheckResult(label, "FAIL", "unexpected response", True)
    return CheckResult(label, "OK", expected_status, True)


def _check_readiness(api_url: str, fetcher: Fetcher) -> CheckResult:
    label = "FastAPI readiness (PostgreSQL)"
    result = fetcher(f"{api_url.rstrip('/')}/ready", CHECK_TIMEOUT_SECONDS)
    if result.status_code != 200:
        database = result.payload.get("database") if isinstance(result.payload, Mapping) else None
        detail = "database unavailable" if database == "unavailable" else _failure_detail(result)
        return CheckResult(label, "FAIL", detail, True)
    if not isinstance(result.payload, Mapping) or result.payload.get("status") != "healthy":
        return CheckResult(label, "FAIL", "unexpected response", True)
    return CheckResult(label, "OK", "database healthy", True)


def _check_streamlit(streamlit_url: str, fetcher: Fetcher) -> CheckResult:
    label = "Streamlit"
    result = fetcher(f"{streamlit_url.rstrip('/')}/_stcore/health", CHECK_TIMEOUT_SECONDS)
    if result.status_code != 200:
        return CheckResult(label, "FAIL", _failure_detail(result), True)
    return CheckResult(label, "OK", "healthy", True)


def _check_ollama(ollama_url: str, model: str, fetcher: Fetcher) -> CheckResult:
    label = "Ollama"
    result = fetcher(f"{ollama_url.rstrip('/')}/api/tags", CHECK_TIMEOUT_SECONDS)
    if result.status_code != 200:
        return CheckResult(label, "WARN", _failure_detail(result), False)
    if not isinstance(result.payload, Mapping):
        return CheckResult(label, "WARN", "malformed response", False)
    models = result.payload.get("models", [])
    names = {
        item.get("name")
        for item in models
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if model not in names:
        return CheckResult(label, "WARN", f"reachable; model '{model}' is not installed", False)
    return CheckResult(label, "OK", f"reachable; model '{model}' available", False)


def _failure_detail(result: HttpResult) -> str:
    if result.status_code is not None:
        return f"HTTP {result.status_code}"
    if result.error_kind == "connection_failure":
        return "connection failed or timed out"
    return "request failed"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("DOCUINTEL_API_BASE_URL", DEFAULT_API_URL))
    parser.add_argument(
        "--streamlit-url",
        default=os.getenv("DOCUINTEL_STREAMLIT_BASE_URL", DEFAULT_STREAMLIT_URL),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print diagnostics and fail only when a required dependency is unhealthy."""

    args = _parser().parse_args(argv)
    checks = run_checks(
        api_url=args.api_url,
        streamlit_url=args.streamlit_url,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )
    print("DocuIntel deployment diagnostics")
    for check in checks:
        requirement = "required" if check.required else "optional"
        print(f"[{check.status}] {check.label} ({requirement}): {check.detail}")

    required_failed = any(check.required and check.status == "FAIL" for check in checks)
    if required_failed:
        print("Result: required deployment dependency failure.")
        return 1
    if any(check.status == "WARN" for check in checks):
        print("Result: core deployment is ready; optional features may be unavailable.")
    else:
        print("Result: deployment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
