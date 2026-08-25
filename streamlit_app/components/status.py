"""Backend status display."""

from __future__ import annotations

from typing import Any

import streamlit as st

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


def render_backend_status(snapshot: dict[str, Any]) -> None:
    health = snapshot.get("health")
    ready = snapshot.get("ready")
    if health:
        st.sidebar.success(f"API online · {health.get('version', 'unknown version')}")
    else:
        st.sidebar.error(f"API unavailable · {snapshot.get('health_error', 'unknown error')}")
    if ready:
        st.sidebar.success("Database ready")
    elif snapshot.get("ready_error"):
        st.sidebar.warning(f"Database not ready · {snapshot['ready_error']}")


def render_system_status(snapshot: dict[str, Any], *, ollama_model: str) -> None:
    """Render a compact main-page status panel without probing external Ollama directly."""

    health = snapshot.get("health")
    ready = snapshot.get("ready")
    with st.container(border=True):
        st.subheader("System status")
        columns = st.columns(3)
        if health:
            columns[0].success("API online")
            columns[0].caption(str(health.get("version", "Version unknown")))
        else:
            columns[0].error("API unavailable")
            columns[0].caption("Start FastAPI, then refresh status.")
        if ready:
            columns[1].success("Database ready")
        else:
            columns[1].warning("Database not ready")
            columns[1].caption("Readiness is reported by FastAPI.")
        columns[2].info("Ollama model")
        columns[2].caption(f"{ollama_model} · checked when an answer is generated")
        st.caption("The UI never starts Ollama and does not access PostgreSQL, models, OCR, or retrieval directly.")
