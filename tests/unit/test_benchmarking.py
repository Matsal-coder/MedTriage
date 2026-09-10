"""Tests for baseline latency benchmarking."""

import json

import pytest

from medtriage.benchmarking.latency import (
    calculate_latency_statistics,
    percentile,
    persist_benchmark,
)

EXPECTED_MEASUREMENTS = 4
EXPECTED_P50 = 2.5
EXPECTED_P95 = 3.85
EXPECTED_MIN = 1.0
EXPECTED_MAX = 4.0


def test_percentile_calculation() -> None:
    """Percentiles should be calculated deterministically."""
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == pytest.approx(EXPECTED_P50)
    assert percentile(values, 0.95) == pytest.approx(EXPECTED_P95)


def test_percentile_rejects_empty_values() -> None:
    """Percentile calculation should reject an empty sequence."""
    with pytest.raises(ValueError, match="empty sequence"):
        percentile([], 0.50)


def test_latency_statistics_return_expected_contract() -> None:
    """Latency statistics should expose the benchmark metric contract."""
    latencies = [1.0, 2.0, 3.0, 4.0]

    results = calculate_latency_statistics(latencies)

    assert results["measured_runs"] == EXPECTED_MEASUREMENTS
    assert results["mean_ms"] == pytest.approx(2.5)
    assert results["p50_ms"] == pytest.approx(EXPECTED_P50)
    assert results["p95_ms"] == pytest.approx(EXPECTED_P95)
    assert results["min_ms"] == EXPECTED_MIN
    assert results["max_ms"] == EXPECTED_MAX
    assert results["throughput_requests_per_second"] > 0


def test_latency_statistics_reject_empty_values() -> None:
    """Latency statistics should reject empty measurements."""
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_latency_statistics([])


def test_benchmark_results_can_be_persisted(tmp_path) -> None:
    """Benchmark results should be serializable and reloadable."""
    results = {
        "benchmark": "baseline_model_inference",
        "mean_ms": 2.5,
    }

    artifact_path = tmp_path / "baseline_latency.json"

    saved_path = persist_benchmark(results, artifact_path)

    with saved_path.open(encoding="utf-8") as file:
        loaded = json.load(file)

    assert saved_path.exists()
    assert loaded == results
