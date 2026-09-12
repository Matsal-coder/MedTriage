"""Equivalence tests between sklearn and ONNX inference backends."""

from pathlib import Path

import numpy as np
import pytest

from medtriage.ci.prepare_model import create_ci_model
from medtriage.modeling.onnx_export import export_model_to_onnx
from medtriage.modeling.onnx_predict import OnnxPredictionService
from medtriage.modeling.predict import PredictionService

PROBABILITY_ABSOLUTE_TOLERANCE = 1e-5

TEST_TEXTS = (
    "urgent emergency patient with severe trauma",
    "critical injury requiring emergency intervention",
    "attention patient with moderate symptoms",
    "clinical evaluation and followup required",
    "normal patient stable routine screening",
    "stable patient attending routine checkup",
)


@pytest.fixture
def equivalent_services(
    tmp_path: Path,
) -> tuple[PredictionService, OnnxPredictionService]:
    """Create sklearn and ONNX services from the same trained model."""
    baseline_path = tmp_path / "baseline.joblib"
    onnx_path = tmp_path / "optimized.onnx"

    create_ci_model(baseline_path)

    export_model_to_onnx(
        model_path=baseline_path,
        artifact_path=onnx_path,
    )

    sklearn_service = PredictionService(baseline_path)
    sklearn_service.load()

    onnx_service = OnnxPredictionService(
        model_path=onnx_path,
        baseline_model_path=baseline_path,
    )
    onnx_service.load()

    return sklearn_service, onnx_service


@pytest.mark.parametrize("text", TEST_TEXTS)
def test_sklearn_and_onnx_predictions_match(
    equivalent_services: tuple[PredictionService, OnnxPredictionService],
    text: str,
) -> None:
    """Both inference backends should return the same predicted class."""
    sklearn_service, onnx_service = equivalent_services

    sklearn_prediction, _, _ = sklearn_service.predict(text)
    onnx_prediction, _, _ = onnx_service.predict(text)

    assert onnx_prediction == sklearn_prediction


@pytest.mark.parametrize("text", TEST_TEXTS)
def test_sklearn_and_onnx_probability_labels_match(
    equivalent_services: tuple[PredictionService, OnnxPredictionService],
    text: str,
) -> None:
    """Both backends should expose probabilities for the same labels."""
    sklearn_service, onnx_service = equivalent_services

    _, sklearn_probabilities, _ = sklearn_service.predict(text)
    _, onnx_probabilities, _ = onnx_service.predict(text)

    assert tuple(sklearn_probabilities) == tuple(onnx_probabilities)


@pytest.mark.parametrize("text", TEST_TEXTS)
def test_sklearn_and_onnx_probabilities_are_close(
    equivalent_services: tuple[PredictionService, OnnxPredictionService],
    text: str,
) -> None:
    """ONNX probabilities should remain numerically close to sklearn."""
    sklearn_service, onnx_service = equivalent_services

    _, sklearn_probabilities, _ = sklearn_service.predict(text)
    _, onnx_probabilities, _ = onnx_service.predict(text)

    sklearn_values = np.array(
        list(sklearn_probabilities.values()),
        dtype=float,
    )
    onnx_values = np.array(
        list(onnx_probabilities.values()),
        dtype=float,
    )

    np.testing.assert_allclose(
        onnx_values,
        sklearn_values,
        rtol=0,
        atol=PROBABILITY_ABSOLUTE_TOLERANCE,
    )


def test_probability_column_order_matches_sklearn_classes(
    equivalent_services: tuple[PredictionService, OnnxPredictionService],
) -> None:
    """ONNX probability ordering should match sklearn class ordering."""
    sklearn_service, onnx_service = equivalent_services

    assert sklearn_service.model is not None

    sklearn_classes = tuple(str(label) for label in sklearn_service.model.classes_)

    _, onnx_probabilities, _ = onnx_service.predict(
        "urgent emergency patient with severe trauma"
    )

    assert tuple(onnx_probabilities) == sklearn_classes
