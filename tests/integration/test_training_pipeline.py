"""Integration tests for the baseline training workflow."""

import joblib
import pandas as pd

from medtriage.modeling.train import run_training

EXPECTED_ROWS = 30


def test_training_pipeline_generates_reloadable_artifact(tmp_path) -> None:
    """The full training workflow should generate a reusable model artifact."""
    data = pd.DataFrame(
        {
            "medical_abstract": [
                f"Urgent oncology cardiovascular case {index}" for index in range(10)
            ]
            + [f"Attention digestive neurological case {index}" for index in range(10)]
            + [f"Normal general pathological case {index}" for index in range(10)],
            "condition_label": [1] * 10 + [2] * 10 + [5] * 10,
        }
    )

    assert len(data) == EXPECTED_ROWS

    dataset_path = tmp_path / "medical_tc_train.csv"
    artifact_path = tmp_path / "baseline_pipeline.joblib"

    data.to_csv(dataset_path, index=False)

    saved_path = run_training(
        dataset_path=dataset_path,
        artifact_path=artifact_path,
    )

    model = joblib.load(saved_path)

    prediction = model.predict(["General pathological routine evaluation"])

    assert saved_path.exists()
    assert prediction[0] in {"normal", "attention", "urgent"}
