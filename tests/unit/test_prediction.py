"""Tests for the prediction service."""

import joblib
import pandas as pd
import pytest

from medtriage.modeling.predict import PredictionService
from medtriage.modeling.train import train_model

EXPECTED_PROBABILITY_TOTAL = 1.0
PROBABILITY_TOLERANCE = 1e-6


def make_training_dataset() -> pd.DataFrame:
    """Create a synthetic dataset for prediction service tests."""
    return pd.DataFrame(
        {
            "medical_abstract": [
                "Cancer tumor oncology treatment",
                "Malignant neoplasm chemotherapy",
                "Cardiac infarction heart disease",
                "Cardiovascular disorder chest pain",
                "Digestive abdominal intestinal disease",
                "Gastrointestinal disorder treatment",
                "Neurological brain nervous disorder",
                "Neurology cognitive symptoms",
                "General pathological condition stable",
                "General clinical finding normal",
                "General condition routine evaluation",
                "General pathology observation",
            ],
            "triage_label": [
                "urgent",
                "urgent",
                "urgent",
                "urgent",
                "attention",
                "attention",
                "attention",
                "attention",
                "normal",
                "normal",
                "normal",
                "normal",
            ],
        }
    )


def test_prediction_service_loads_and_predicts(tmp_path) -> None:
    """The service should load a persisted model and return prediction data."""
    model = train_model(make_training_dataset())

    model_path = tmp_path / "baseline_pipeline.joblib"
    joblib.dump(model, model_path)

    service = PredictionService(model_path)
    service.load()

    prediction, probabilities, inference_time_ms = service.predict(
        "Cardiovascular patient with acute symptoms"
    )

    assert prediction in {"normal", "attention", "urgent"}
    assert set(probabilities) == {"attention", "normal", "urgent"}
    assert sum(probabilities.values()) == pytest.approx(
        EXPECTED_PROBABILITY_TOTAL,
        abs=PROBABILITY_TOLERANCE,
    )
    assert inference_time_ms >= 0


def test_prediction_service_rejects_prediction_before_load(tmp_path) -> None:
    """Prediction should fail when the model has not been loaded."""
    service = PredictionService(tmp_path / "baseline_pipeline.joblib")

    with pytest.raises(RuntimeError, match="has not been loaded"):
        service.predict("Example medical text")


def test_prediction_service_rejects_missing_model(tmp_path) -> None:
    """Loading should fail clearly when the artifact does not exist."""
    service = PredictionService(tmp_path / "missing.joblib")

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        service.load()
