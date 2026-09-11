"""Tests for the temporary CI model artifact."""

from medtriage.ci.prepare_model import create_ci_model
from medtriage.config import TRIAGE_CLASSES
from medtriage.modeling.predict import PredictionService


def test_ci_model_is_runtime_compatible(tmp_path):
    """Ensure the CI artifact satisfies the prediction runtime contract."""
    artifact_path = tmp_path / "baseline_pipeline.joblib"

    created_path = create_ci_model(artifact_path)

    service = PredictionService(model_path=created_path)
    service.load()

    prediction, probabilities, inference_time_ms = service.predict(
        "urgent emergency patient"
    )

    assert created_path.exists()
    assert prediction in TRIAGE_CLASSES
    assert set(probabilities) == set(TRIAGE_CLASSES)
    assert inference_time_ms >= 0
