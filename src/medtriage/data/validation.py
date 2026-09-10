"""Dataset validation utilities."""

import pandas as pd

from medtriage.config import ORIGINAL_TARGET_COLUMN, TEXT_COLUMN

VALID_CONDITION_LABELS = frozenset({1, 2, 3, 4, 5})


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the minimum contract required by the medical dataset."""
    if data.empty:
        raise ValueError("Dataset must not be empty.")

    required_columns = {TEXT_COLUMN, ORIGINAL_TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if data[TEXT_COLUMN].isna().any():
        raise ValueError("Dataset contains null medical abstracts.")

    if data[ORIGINAL_TARGET_COLUMN].isna().any():
        raise ValueError("Dataset contains null condition labels.")

    empty_text = data[TEXT_COLUMN].astype(str).str.strip().eq("")
    if empty_text.any():
        raise ValueError("Dataset contains empty medical abstracts.")

    labels = set(data[ORIGINAL_TARGET_COLUMN].unique())
    invalid_labels = labels.difference(VALID_CONDITION_LABELS)

    if invalid_labels:
        invalid = ", ".join(str(label) for label in sorted(invalid_labels))
        raise ValueError(f"Dataset contains invalid condition labels: {invalid}")
