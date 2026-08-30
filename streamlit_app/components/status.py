"""Backend status display."""

from __future__ import annotations

from typing import Any

from streamlit_app.api.client import ApiClient, ApiError


def get_backend_status(client: ApiClient) -> dict[str, Any]:
    """Fetch both liveness and readiness through the public API."""

    snapshot: dict[str, Any] = {}
    try:
        snapshot["health"] = client.get("/health")
    except ApiError as exc:
        snapshot["health_error"] = str(exc)
    try:
        snapshot["ready"] = client.get("/ready")
    except ApiError as exc:
        snapshot["ready_error"] = str(exc)
    return snapshot


def render_backend_status(_snapshot: dict[str, Any]) -> None:
    """Retain the old presentation import without rendering operational status."""

    return None


def render_system_status(_snapshot: dict[str, Any], *, ollama_model: str) -> None:
    """Retain the old presentation import without rendering a system-status panel."""

    del ollama_model
    return None
