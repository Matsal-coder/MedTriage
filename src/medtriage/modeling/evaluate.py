"""Baseline model evaluation utilities."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from medtriage.config import (
    EVALUATION_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    TEST_DATA_PATH,
    TEXT_COLUMN,
    TRAIN_DATA_PATH,
    TRIAGE_CLASSES,
    TRIAGE_TARGET_COLUMN,
)
from medtriage.data.loader import (
    add_triage_labels,
    load_dataset,
    split_training_data,
)


def calculate_metrics(
    y_true: pd.Series,
    y_pred: list[str] | pd.Series,
) -> dict[str, Any]:
    """Calculate classification metrics for the triage task."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=TRIAGE_CLASSES,
        average=None,
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(TRIAGE_CLASSES)
    }

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=TRIAGE_CLASSES,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "urgent_recall": per_class["urgent"]["recall"],
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(TRIAGE_CLASSES),
        "evaluated_samples": len(y_true),
    }


def evaluate_model(
    model: Any,
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate a trained model against a prepared dataset."""
    predictions = model.predict(data[TEXT_COLUMN])

    return calculate_metrics(
        data[TRIAGE_TARGET_COLUMN],
        predictions,
    )


def persist_evaluation(
    results: dict[str, Any],
    artifact_path: Path = EVALUATION_ARTIFACT_PATH,
) -> Path:
    """Persist evaluation results as JSON."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return artifact_path


def run_evaluation(
    model_path: Path = MODEL_ARTIFACT_PATH,
    train_dataset_path: Path = TRAIN_DATA_PATH,
    test_dataset_path: Path = TEST_DATA_PATH,
    artifact_path: Path = EVALUATION_ARTIFACT_PATH,
) -> Path:
    """Run validation and official test evaluation."""
    model = joblib.load(model_path)

    train_raw = load_dataset(train_dataset_path)
    train_prepared = add_triage_labels(train_raw)

    _, validation_data = split_training_data(train_prepared)

    test_raw = load_dataset(test_dataset_path)
    test_data = add_triage_labels(test_raw)

    results = {
        "validation": evaluate_model(model, validation_data),
        "test": evaluate_model(model, test_data),
    }

    return persist_evaluation(results, artifact_path)


def main() -> None:
    """Run baseline evaluation from the command line."""
    artifact_path = run_evaluation()
    print(f"Evaluation artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
