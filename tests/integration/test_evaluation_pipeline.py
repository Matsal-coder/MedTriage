"""Integration tests for the baseline evaluation workflow."""

import json

import pandas as pd

from medtriage.modeling.evaluate import run_evaluation
from medtriage.modeling.train import run_training


def make_dataset(rows_per_group: int) -> pd.DataFrame:
    """Create a synthetic dataset using original corpus labels."""
    return pd.DataFrame(
        {
            "medical_abstract": (
                [
                    f"Oncology malignant neoplasm urgent case {index}"
                    for index in range(rows_per_group)
                ]
                + [
                    f"Digestive abdominal attention case {index}"
                    for index in range(rows_per_group)
                ]
                + [
                    f"General pathological normal case {index}"
                    for index in range(rows_per_group)
                ]
            ),
            "condition_label": (
                [1] * rows_per_group + [2] * rows_per_group + [5] * rows_per_group
            ),
        }
    )


def test_evaluation_pipeline_generates_metrics_artifact(tmp_path) -> None:
    """The full evaluation workflow should persist validation and test metrics."""
    train_data = make_dataset(rows_per_group=10)
    test_data = make_dataset(rows_per_group=5)

    train_path = tmp_path / "medical_tc_train.csv"
    test_path = tmp_path / "medical_tc_test.csv"
    model_path = tmp_path / "baseline_pipeline.joblib"
    evaluation_path = tmp_path / "evaluation.json"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    run_training(
        dataset_path=train_path,
        artifact_path=model_path,
    )

    saved_path = run_evaluation(
        model_path=model_path,
        train_dataset_path=train_path,
        test_dataset_path=test_path,
        artifact_path=evaluation_path,
    )

    with saved_path.open(encoding="utf-8") as file:
        results = json.load(file)

    assert saved_path.exists()
    assert set(results) == {"validation", "test"}

    assert results["validation"]["evaluated_samples"] > 0
    assert results["test"]["evaluated_samples"] > 0

    assert "f1_macro" in results["validation"]
    assert "f1_macro" in results["test"]

    assert "urgent_recall" in results["validation"]
    assert "urgent_recall" in results["test"]
