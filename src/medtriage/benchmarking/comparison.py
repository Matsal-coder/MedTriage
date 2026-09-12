"""Latency comparison between baseline and optimized inference backends."""

import json
from pathlib import Path
from typing import Any

from medtriage.benchmarking.latency import (
    run_benchmark,
    run_optimized_benchmark,
)
from medtriage.config import (
    BASELINE_BENCHMARK_PATH,
    LATENCY_COMPARISON_PATH,
    OPTIMIZED_BENCHMARK_PATH,
)

COMPARISON_FIELDS = (
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "throughput_requests_per_second",
)


def load_benchmark(
    artifact_path: Path,
) -> dict[str, Any]:
    """Load a persisted benchmark artifact."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Benchmark artifact not found: {artifact_path}")

    with artifact_path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_comparable_benchmarks(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> None:
    """Validate that benchmark methodologies are directly comparable."""
    comparable_fields = (
        "latency_scope",
        "warmup_runs",
        "measured_runs",
        "inputs_count",
    )

    for field in comparable_fields:
        if baseline[field] != optimized[field]:
            raise ValueError(
                f"Benchmark mismatch for '{field}': "
                f"{baseline[field]!r} != {optimized[field]!r}"
            )


def calculate_speedup(
    baseline_latency_ms: float,
    optimized_latency_ms: float,
) -> float:
    """Calculate latency speedup as baseline divided by optimized."""
    if optimized_latency_ms <= 0:
        raise ValueError("Optimized latency must be greater than zero.")

    return baseline_latency_ms / optimized_latency_ms


def build_latency_comparison(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    """Build the final latency comparison artifact."""
    validate_comparable_benchmarks(
        baseline,
        optimized,
    )

    metrics = {
        field: {
            "baseline": baseline[field],
            "optimized": optimized[field],
        }
        for field in COMPARISON_FIELDS
    }

    return {
        "latency_scope": baseline["latency_scope"],
        "warmup_runs": baseline["warmup_runs"],
        "measured_runs": baseline["measured_runs"],
        "inputs_count": baseline["inputs_count"],
        "baseline_benchmark": baseline["benchmark"],
        "optimized_benchmark": optimized["benchmark"],
        "metrics": metrics,
        "mean_speedup": calculate_speedup(
            baseline["mean_ms"],
            optimized["mean_ms"],
        ),
        "p50_speedup": calculate_speedup(
            baseline["p50_ms"],
            optimized["p50_ms"],
        ),
        "p95_speedup": calculate_speedup(
            baseline["p95_ms"],
            optimized["p95_ms"],
        ),
    }


def persist_latency_comparison(
    results: dict[str, Any],
    artifact_path: Path = LATENCY_COMPARISON_PATH,
) -> Path:
    """Persist the latency comparison as JSON."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return artifact_path


def run_latency_comparison(
    baseline_path: Path = BASELINE_BENCHMARK_PATH,
    optimized_path: Path = OPTIMIZED_BENCHMARK_PATH,
    comparison_path: Path = LATENCY_COMPARISON_PATH,
) -> Path:
    """Compare persisted baseline and optimized benchmark artifacts."""
    baseline = load_benchmark(baseline_path)
    optimized = load_benchmark(optimized_path)

    results = build_latency_comparison(
        baseline,
        optimized,
    )

    return persist_latency_comparison(
        results,
        comparison_path,
    )


def main() -> None:
    """Benchmark both backends and generate their latency comparison."""

    baseline_path = run_benchmark()
    print(f"Baseline benchmark saved to: {baseline_path}")

    optimized_path = run_optimized_benchmark()
    print(f"Optimized benchmark saved to: {optimized_path}")

    comparison_path = run_latency_comparison()
    print(f"Latency comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
