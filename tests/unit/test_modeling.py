"""Tests for baseline model training."""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from medtriage.config import TRIAGE_TARGET_COLUMN
from medtriage.modeling.train import (
    build_model_pipeline,
    persist_model,
    train_model,
)

EXPECTED_CLASSES = {"attention", "normal", "urgent"}


def make_training_dataset() -> pd.DataFrame:
    """Create a small synthetic dataset suitable for model training."""
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
            TRIAGE_TARGET_COLUMN: [
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


def test_build_model_pipeline() -> None:
    """The baseline model should use a scikit-learn pipeline."""
    model = build_model_pipeline()

    assert isinstance(model, Pipeline)
    assert set(model.named_steps) == {"tfidf", "classifier"}


def test_model_can_be_trained() -> None:
    """The baseline pipeline should train on a valid prepared dataset."""
    training_data = make_training_dataset()

    model = train_model(training_data)

    predictions = model.predict(training_data["medical_abstract"])

    assert len(predictions) == len(training_data)
    assert set(predictions).issubset(EXPECTED_CLASSES)


def test_model_can_be_persisted_and_reloaded(tmp_path) -> None:
    """A persisted model should remain usable after reloading."""

    training_data = make_training_dataset()
    model = train_model(training_data)

    artifact_path = tmp_path / "baseline_pipeline.joblib"
    saved_path = persist_model(model, artifact_path)

    reloaded_model = joblib.load(saved_path)
    predictions = reloaded_model.predict(training_data["medical_abstract"])

    assert saved_path.exists()
    assert len(predictions) == len(training_data)
    assert set(predictions).issubset(EXPECTED_CLASSES)
