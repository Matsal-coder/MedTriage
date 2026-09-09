"""Integration tests for the health endpoint."""

from fastapi import status
from fastapi.testclient import TestClient

from medtriage.api.app import app

client = TestClient(app)


def test_application_can_be_imported() -> None:
    """FastAPI application should be available."""
    assert app is not None


def test_health_returns_http_200() -> None:
    """Health endpoint should return HTTP 200."""
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK


def test_health_returns_expected_schema() -> None:
    """Health endpoint should return the expected response contract."""
    response = client.get("/health")

    assert response.json() == {"status": "ok"}
