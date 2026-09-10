"""Integration tests for the data preparation pipeline."""

import pandas as pd

from medtriage.config import TRIAGE_TARGET_COLUMN
from medtriage.data.loader import (
    add_triage_labels,
    load_dataset,
    split_training_data,
)

EXPECTED_TOTAL_ROWS = 15
EXPECTED_TRAIN_ROWS = 12
EXPECTED_VALIDATION_ROWS = 3


def test_dataset_can_be_loaded_and_prepared(tmp_path) -> None:
    """The data pipeline should load, map, and split a valid dataset."""
    data = pd.DataFrame(
        {
            "medical_abstract": [
                "Neoplasm example one",
                "Neoplasm example two",
                "Neoplasm example three",
                "Cardiovascular example one",
                "Cardiovascular example two",
                "Digestive example one",
                "Digestive example two",
                "Digestive example three",
                "Nervous example one",
                "Nervous example two",
                "General condition one",
                "General condition two",
                "General condition three",
                "General condition four",
                "General condition five",
            ],
            "condition_label": [
                1,
                1,
                1,
                4,
                4,
                2,
                2,
                2,
                3,
                3,
                5,
                5,
                5,
                5,
                5,
            ],
        }
    )

    dataset_path = tmp_path / "medical_tc_train.csv"
    data.to_csv(dataset_path, index=False)

    loaded = load_dataset(dataset_path)
    prepared = add_triage_labels(loaded)

    assert len(prepared) == EXPECTED_TOTAL_ROWS
    assert TRIAGE_TARGET_COLUMN in prepared.columns
    assert set(prepared[TRIAGE_TARGET_COLUMN]) == {
        "normal",
        "attention",
        "urgent",
    }

    train_data, validation_data = split_training_data(prepared)

    assert len(train_data) == EXPECTED_TRAIN_ROWS
    assert len(validation_data) == EXPECTED_VALIDATION_ROWS

    assert set(train_data[TRIAGE_TARGET_COLUMN]) == {
        "normal",
        "attention",
        "urgent",
    }
    assert set(validation_data[TRIAGE_TARGET_COLUMN]) == {
        "normal",
        "attention",
        "urgent",
    }


def test_training_split_is_reproducible(tmp_path) -> None:
    """The same dataset and seed should always produce the same split."""
    data = pd.DataFrame(
        {
            "medical_abstract": [f"Urgent example {index}" for index in range(5)]
            + [f"Attention example {index}" for index in range(5)]
            + [f"Normal example {index}" for index in range(5)],
            "condition_label": [1] * 5 + [2] * 5 + [5] * 5,
        }
    )

    dataset_path = tmp_path / "medical_tc_train.csv"
    data.to_csv(dataset_path, index=False)

    prepared = add_triage_labels(load_dataset(dataset_path))

    first_train, first_validation = split_training_data(prepared)
    second_train, second_validation = split_training_data(prepared)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_validation, second_validation)
