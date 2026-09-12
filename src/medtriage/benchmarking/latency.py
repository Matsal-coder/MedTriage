"""Latency benchmarking utilities for MedTriage inference backends."""

import json
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from medtriage.config import (
    BASELINE_BENCHMARK_PATH,
    BENCHMARK_MEASURED_RUNS,
    BENCHMARK_WARMUP_RUNS,
    MODEL_ARTIFACT_PATH,
    OPTIMIZED_BENCHMARK_PATH,
    OPTIMIZED_MODEL_ARTIFACT_PATH,
)
from medtriage.modeling.onnx_predict import OnnxPredictionService
from medtriage.modeling.predict import PredictionService

BENCHMARK_TEXTS = (
    "Patient with cardiovascular symptoms and chest pain.",
    "Patient undergoing evaluation for digestive system symptoms.",
    "General pathological condition under routine clinical assessment.",
    "Patient with neurological symptoms requiring medical evaluation.",
    "Patient with suspected neoplasm undergoing diagnostic investigation.",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration required to run a latency benchmark."""

    benchmark_name: str
    model_artifact: Path
    artifact_path: Path
    warmup_runs: int = BENCHMARK_WARMUP_RUNS
    measured_runs: int = BENCHMARK_MEASURED_RUNS


class PredictionBackend(Protocol):
    """Minimal prediction contract required by the latency benchmark."""

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Return prediction, probabilities, and inference latency."""


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate percentile using linear interpolation."""
    ordered = sorted(values)

    if not ordered:
        raise ValueError("Cannot calculate percentile of an empty sequence.")

    position = (len(ordered) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    fraction = position - lower_index

    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def calculate_latency_statistics(
    latencies_ms: list[float],
) -> dict[str, float | int]:
    """Calculate latency and throughput statistics."""
    if not latencies_ms:
        raise ValueError("Latency measurements must not be empty.")

    total_seconds = sum(latencies_ms) / 1000

    return {
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput_requests_per_second": len(latencies_ms) / total_seconds,
        "measured_runs": len(latencies_ms),
    }


def persist_benchmark(
    results: dict[str, Any],
    artifact_path: Path = BASELINE_BENCHMARK_PATH,
) -> Path:
    """Persist benchmark results as JSON."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return artifact_path


def benchmark_backend(
    service: PredictionBackend,
    config: BenchmarkConfig,
) -> Path:
    """Benchmark a loaded prediction backend using the official methodology."""
    for index in range(config.warmup_runs):
        text = BENCHMARK_TEXTS[index % len(BENCHMARK_TEXTS)]
        service.predict(text)

    latencies_ms: list[float] = []

    for index in range(config.measured_runs):
        text = BENCHMARK_TEXTS[index % len(BENCHMARK_TEXTS)]

        _, _, inference_time_ms = service.predict(text)

        latencies_ms.append(inference_time_ms)

    statistics_result = calculate_latency_statistics(latencies_ms)

    results = {
        "benchmark": config.benchmark_name,
        "model_artifact": str(config.model_artifact),
        "latency_scope": "model_inference",
        "warmup_runs": config.warmup_runs,
        "inputs_count": len(BENCHMARK_TEXTS),
        **statistics_result,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }

    return persist_benchmark(
        results,
        config.artifact_path,
    )


def run_benchmark(
    model_path: Path = MODEL_ARTIFACT_PATH,
    artifact_path: Path = BASELINE_BENCHMARK_PATH,
    warmup_runs: int = BENCHMARK_WARMUP_RUNS,
    measured_runs: int = BENCHMARK_MEASURED_RUNS,
) -> Path:
    """Run the reproducible baseline model latency benchmark."""
    service = PredictionService(model_path)
    service.load()

    config = BenchmarkConfig(
        benchmark_name="baseline_model_inference",
        model_artifact=model_path,
        artifact_path=artifact_path,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    return benchmark_backend(
        service,
        config,
    )


def run_optimized_benchmark(
    optimized_model_path: Path = OPTIMIZED_MODEL_ARTIFACT_PATH,
    baseline_model_path: Path = MODEL_ARTIFACT_PATH,
    artifact_path: Path = OPTIMIZED_BENCHMARK_PATH,
    warmup_runs: int = BENCHMARK_WARMUP_RUNS,
    measured_runs: int = BENCHMARK_MEASURED_RUNS,
) -> Path:
    """Run the optimized ONNX model latency benchmark."""
    service = OnnxPredictionService(
        model_path=optimized_model_path,
        baseline_model_path=baseline_model_path,
    )
    service.load()

    config = BenchmarkConfig(
        benchmark_name="optimized_onnx_model_inference",
        model_artifact=optimized_model_path,
        artifact_path=artifact_path,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    return benchmark_backend(
        service,
        config,
    )


def main() -> None:
    """Run the baseline latency benchmark from the command line."""
    artifact_path = run_benchmark()
    print(f"Benchmark artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
