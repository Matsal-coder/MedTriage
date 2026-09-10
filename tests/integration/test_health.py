"""Integration tests for the health endpoint."""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from medtriage.api.app import app


class FakePredictionService:
    """Prediction service used to isolate health endpoint tests."""

    def load(self) -> None:
        """Simulate successful model loading."""

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Return a deterministic prediction if called."""
        return (
            "normal",
            {
                "attention": 0.1,
                "normal": 0.8,
                "urgent": 0.1,
            },
            1.0,
        )


def test_application_can_be_imported() -> None:
    """FastAPI application should be available."""
    assert app is not None


def test_health_returns_http_200(monkeypatch: Any) -> None:
    """Health endpoint should return HTTP 200."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK


def test_health_returns_expected_schema(monkeypatch: Any) -> None:
    """Health endpoint should return the expected response contract."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.json() == {"status": "ok"}
