"""Tests for the ONNX prediction service."""

from pathlib import Path

import pytest

from medtriage.ci.prepare_model import create_ci_model
from medtriage.modeling.onnx_export import export_model_to_onnx
from medtriage.modeling.onnx_predict import OnnxPredictionService

EXPECTED_PROBABILITY_TOTAL = 1.0
PROBABILITY_TOLERANCE = 1e-5


def create_test_models(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Create matching sklearn and ONNX artifacts for prediction tests."""
    baseline_path = tmp_path / "baseline.joblib"
    onnx_path = tmp_path / "optimized.onnx"

    create_ci_model(baseline_path)

    export_model_to_onnx(
        model_path=baseline_path,
        artifact_path=onnx_path,
    )

    return baseline_path, onnx_path


def test_onnx_prediction_service_loads_and_predicts(
    tmp_path: Path,
) -> None:
    """The ONNX service should load and return prediction data."""
    baseline_path, onnx_path = create_test_models(tmp_path)

    service = OnnxPredictionService(
        model_path=onnx_path,
        baseline_model_path=baseline_path,
    )
    service.load()

    prediction, probabilities, inference_time_ms = service.predict(
        "urgent emergency patient with severe trauma"
    )

    assert prediction in {"normal", "attention", "urgent"}
    assert set(probabilities) == {"attention", "normal", "urgent"}
    assert sum(probabilities.values()) == pytest.approx(
        EXPECTED_PROBABILITY_TOTAL,
        abs=PROBABILITY_TOLERANCE,
    )
    assert inference_time_ms >= 0


def test_onnx_prediction_service_rejects_prediction_before_load(
    tmp_path: Path,
) -> None:
    """Prediction should fail when the ONNX model has not been loaded."""
    service = OnnxPredictionService(tmp_path / "optimized.onnx")

    with pytest.raises(RuntimeError, match="has not been loaded"):
        service.predict("Example medical text")


def test_onnx_prediction_service_rejects_missing_model(
    tmp_path: Path,
) -> None:
    """Loading should fail clearly when the ONNX artifact does not exist."""
    baseline_path = tmp_path / "baseline.joblib"
    create_ci_model(baseline_path)

    service = OnnxPredictionService(
        model_path=tmp_path / "missing.onnx",
        baseline_model_path=baseline_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="ONNX model artifact not found",
    ):
        service.load()
