"""Integration tests for the prediction endpoint."""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from medtriage.api.app import app

EXPECTED_INFERENCE_TIME_MS = 1.5


class FakePredictionService:
    """Small deterministic prediction service used by API tests."""

    def load(self) -> None:
        """Simulate model loading."""

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Return deterministic prediction results."""
        return (
            "urgent",
            {
                "attention": 0.1,
                "normal": 0.2,
                "urgent": 0.7,
            },
            1.5,
        )


def test_predict_returns_expected_response(monkeypatch: Any) -> None:
    """A valid request should return the prediction contract."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Patient with acute cardiovascular symptoms."},
        )

    assert response.status_code == status.HTTP_200_OK

    body = response.json()

    assert body["prediction"] == "urgent"
    assert body["probabilities"] == {
        "attention": 0.1,
        "normal": 0.2,
        "urgent": 0.7,
    }
    assert body["inference_time_ms"] == EXPECTED_INFERENCE_TIME_MS


def test_predict_rejects_empty_text(monkeypatch: Any) -> None:
    """An empty medical text should be rejected."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": ""},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_rejects_whitespace_only_text(monkeypatch: Any) -> None:
    """Whitespace-only text should be rejected."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "   "},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
