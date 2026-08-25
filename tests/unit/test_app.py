"""Tests for the Module 0 FastAPI application."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app


def test_application_creation() -> None:
    """The application factory returns a configured FastAPI instance."""

    application = create_app()

    assert isinstance(application, FastAPI)
    assert application.title == "DocuIntel"
    assert application.version == "0.1.0"


def test_health_returns_http_200() -> None:
    """The health endpoint responds successfully."""

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_health_response_structure() -> None:
    """The health endpoint exposes the documented response fields."""

    with TestClient(create_app()) as client:
        payload = client.get("/health").json()

    assert payload == {
        "status": "healthy",
        "service": "DocuIntel",
        "version": "0.1.0",
    }
