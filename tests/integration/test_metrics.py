"""Integration tests for Prometheus HTTP metrics."""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY

from medtriage.api.app import app


class FakePredictionService:
    """Small deterministic prediction service used by metrics tests."""

    def load(self) -> None:
        """Simulate model loading."""

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Return deterministic prediction results."""
        return (
            "normal",
            {
                "attention": 0.1,
                "normal": 0.8,
                "urgent": 0.1,
            },
            1.0,
        )


def get_sample_value(
    metric_name: str,
    labels: dict[str, str],
) -> float:
    """Return a Prometheus sample value or zero when it does not exist yet."""
    value = REGISTRY.get_sample_value(metric_name, labels)

    if value is None:
        return 0.0

    return value


def get_requests_total() -> float:
    """Return the sum of every MedTriage HTTP request counter sample."""
    metric = next(
        collector
        for collector in REGISTRY.collect()
        if collector.name == "medtriage_http_requests"
    )

    return sum(
        sample.value
        for sample in metric.samples
        if sample.name == "medtriage_http_requests_total"
    )


def test_metrics_endpoint_returns_prometheus_payload(
    monkeypatch: Any,
) -> None:
    """Metrics endpoint should expose Prometheus text format."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert "medtriage_http_requests_total" in response.text
    assert "medtriage_http_request_duration_seconds" in response.text
    assert "medtriage_http_errors_total" in response.text


def test_health_request_increments_request_counter(
    monkeypatch: Any,
) -> None:
    """A health request should increment its HTTP request counter."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    labels = {
        "method": "GET",
        "route": "/health",
        "status_code": "200",
    }

    before = get_sample_value(
        "medtriage_http_requests_total",
        labels,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    after = get_sample_value(
        "medtriage_http_requests_total",
        labels,
    )

    assert response.status_code == status.HTTP_200_OK
    assert after == before + 1


def test_health_request_records_duration(
    monkeypatch: Any,
) -> None:
    """A health request should add an observation to the latency histogram."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    labels = {
        "method": "GET",
        "route": "/health",
    }

    before = get_sample_value(
        "medtriage_http_request_duration_seconds_count",
        labels,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    after = get_sample_value(
        "medtriage_http_request_duration_seconds_count",
        labels,
    )

    assert response.status_code == status.HTTP_200_OK
    assert after == before + 1


def test_validation_error_increments_error_counter(
    monkeypatch: Any,
) -> None:
    """A response with status >= 400 should increment the error counter."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    request_labels = {
        "method": "POST",
        "route": "/predict",
        "status_code": "422",
    }
    error_labels = {
        "method": "POST",
        "route": "/predict",
        "status_code": "422",
    }

    requests_before = get_sample_value(
        "medtriage_http_requests_total",
        request_labels,
    )
    errors_before = get_sample_value(
        "medtriage_http_errors_total",
        error_labels,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": ""},
        )

    requests_after = get_sample_value(
        "medtriage_http_requests_total",
        request_labels,
    )
    errors_after = get_sample_value(
        "medtriage_http_errors_total",
        error_labels,
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert requests_after == requests_before + 1
    assert errors_after == errors_before + 1


def test_metrics_endpoint_is_excluded_from_instrumentation(
    monkeypatch: Any,
) -> None:
    """Scraping metrics should not change the HTTP request counter."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    before = get_requests_total()

    with TestClient(app) as client:
        response = client.get("/metrics")

    after = get_requests_total()

    assert response.status_code == status.HTTP_200_OK
    assert after == before


def test_documentation_routes_are_excluded_from_instrumentation(
    monkeypatch: Any,
) -> None:
    """Swagger and OpenAPI routes should not affect application metrics."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    before = get_requests_total()

    with TestClient(app) as client:
        docs_response = client.get("/docs")
        openapi_response = client.get("/openapi.json")

    after = get_requests_total()

    assert docs_response.status_code == status.HTTP_200_OK
    assert openapi_response.status_code == status.HTTP_200_OK
    assert after == before


def test_unknown_route_uses_bounded_label_and_records_error(
    monkeypatch: Any,
) -> None:
    """Unknown URLs should share a bounded route label instead of raw paths."""
    monkeypatch.setattr(
        "medtriage.api.app.create_prediction_service",
        FakePredictionService,
    )

    request_labels = {
        "method": "GET",
        "route": "__unmatched__",
        "status_code": "404",
    }
    error_labels = {
        "method": "GET",
        "route": "__unmatched__",
        "status_code": "404",
    }

    requests_before = get_sample_value(
        "medtriage_http_requests_total",
        request_labels,
    )
    errors_before = get_sample_value(
        "medtriage_http_errors_total",
        error_labels,
    )

    with TestClient(app) as client:
        response = client.get("/route-that-does-not-exist/12345")

    requests_after = get_sample_value(
        "medtriage_http_requests_total",
        request_labels,
    )
    errors_after = get_sample_value(
        "medtriage_http_errors_total",
        error_labels,
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert requests_after == requests_before + 1
    assert errors_after == errors_before + 1
