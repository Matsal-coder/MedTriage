"""Tests for baseline model evaluation."""

import json

import pandas as pd

from medtriage.modeling.evaluate import (
    calculate_metrics,
    persist_evaluation,
)

EXPECTED_SAMPLES = 6
EXPECTED_CLASS_COUNT = 3


def test_calculate_metrics_returns_expected_structure() -> None:
    """Evaluation should return the complete metric contract."""
    y_true = pd.Series(
        [
            "urgent",
            "urgent",
            "attention",
            "attention",
            "normal",
            "normal",
        ]
    )

    y_pred = [
        "urgent",
        "attention",
        "attention",
        "attention",
        "normal",
        "urgent",
    ]

    metrics = calculate_metrics(y_true, y_pred)

    assert set(metrics) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "urgent_recall",
        "per_class",
        "confusion_matrix",
        "confusion_matrix_labels",
        "evaluated_samples",
    }

    assert metrics["evaluated_samples"] == EXPECTED_SAMPLES
    assert set(metrics["per_class"]) == {
        "attention",
        "normal",
        "urgent",
    }

    assert len(metrics["confusion_matrix"]) == EXPECTED_CLASS_COUNT
    assert all(len(row) == EXPECTED_CLASS_COUNT for row in metrics["confusion_matrix"])


def test_perfect_predictions_generate_perfect_scores() -> None:
    """Perfect predictions should generate perfect aggregate metrics."""
    y_true = pd.Series(
        [
            "urgent",
            "attention",
            "normal",
        ]
    )

    metrics = calculate_metrics(y_true, y_true.tolist())

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert metrics["urgent_recall"] == 1.0


def test_evaluation_results_can_be_persisted(tmp_path) -> None:
    """Evaluation metrics should be serializable and reloadable."""
    results = {
        "validation": {
            "accuracy": 0.8,
        },
        "test": {
            "accuracy": 0.75,
        },
    }

    artifact_path = tmp_path / "evaluation.json"

    saved_path = persist_evaluation(results, artifact_path)

    with saved_path.open(encoding="utf-8") as file:
        loaded = json.load(file)

    assert saved_path.exists()
    assert loaded == results
