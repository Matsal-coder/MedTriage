"""Tests for baseline latency benchmarking."""

import json

import pytest

from medtriage.benchmarking.comparison import (
    build_latency_comparison,
    calculate_speedup,
    validate_comparable_benchmarks,
)
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


def test_speedup_calculation() -> None:
    """Speedup should divide baseline latency by optimized latency."""
    assert calculate_speedup(4.0, 2.0) == pytest.approx(2.0)


def test_speedup_rejects_non_positive_optimized_latency() -> None:
    """Speedup calculation should reject invalid optimized latency."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_speedup(4.0, 0.0)


def test_comparable_benchmarks_are_validated() -> None:
    """Matching benchmark methodologies should be accepted."""
    baseline = {
        "latency_scope": "model_inference",
        "warmup_runs": 20,
        "measured_runs": 500,
        "inputs_count": 5,
    }
    optimized = baseline.copy()

    validate_comparable_benchmarks(
        baseline,
        optimized,
    )


def test_mismatched_benchmarks_are_rejected() -> None:
    """Different benchmark methodologies should not be compared."""
    baseline = {
        "latency_scope": "model_inference",
        "warmup_runs": 20,
        "measured_runs": 500,
        "inputs_count": 5,
    }
    optimized = {
        **baseline,
        "measured_runs": 100,
    }

    with pytest.raises(
        ValueError,
        match="measured_runs",
    ):
        validate_comparable_benchmarks(
            baseline,
            optimized,
        )


def test_latency_comparison_contains_speedups() -> None:
    """Comparison should expose baseline, optimized, and speedup metrics."""
    baseline = {
        "benchmark": "baseline_model_inference",
        "latency_scope": "model_inference",
        "warmup_runs": 20,
        "measured_runs": 500,
        "inputs_count": 5,
        "mean_ms": 4.0,
        "p50_ms": 3.0,
        "p95_ms": 5.0,
        "min_ms": 2.0,
        "max_ms": 6.0,
        "throughput_requests_per_second": 250.0,
    }

    optimized = {
        "benchmark": "optimized_onnx_model_inference",
        "latency_scope": "model_inference",
        "warmup_runs": 20,
        "measured_runs": 500,
        "inputs_count": 5,
        "mean_ms": 2.0,
        "p50_ms": 1.5,
        "p95_ms": 2.5,
        "min_ms": 1.0,
        "max_ms": 3.0,
        "throughput_requests_per_second": 500.0,
    }

    comparison = build_latency_comparison(
        baseline,
        optimized,
    )

    assert comparison["mean_speedup"] == pytest.approx(2.0)
    assert comparison["p50_speedup"] == pytest.approx(2.0)
    assert comparison["p95_speedup"] == pytest.approx(2.0)

    assert comparison["metrics"]["mean_ms"] == {
        "baseline": 4.0,
        "optimized": 2.0,
    }
