"""Medical Abstracts TC Corpus loading and preparation utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from medtriage.config import (
    ORIGINAL_TARGET_COLUMN,
    RANDOM_SEED,
    TEXT_COLUMN,
    TRIAGE_TARGET_COLUMN,
    VALIDATION_SIZE,
)
from medtriage.data.validation import validate_dataset

TRIAGE_LABEL_MAPPING = {
    1: "urgent",
    2: "attention",
    3: "attention",
    4: "urgent",
    5: "normal",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a Medical Abstracts TC Corpus CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.read_csv(path)
    validate_dataset(data)

    return data


def add_triage_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Add the academic triage proxy while preserving the original target."""
    prepared = data.copy()

    prepared[TEXT_COLUMN] = (
        prepared[TEXT_COLUMN]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    prepared[TRIAGE_TARGET_COLUMN] = prepared[ORIGINAL_TARGET_COLUMN].map(
        TRIAGE_LABEL_MAPPING
    )

    return prepared


def split_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into reproducible stratified train and validation sets."""
    train_data, validation_data = train_test_split(
        data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=data[TRIAGE_TARGET_COLUMN],
    )

    return train_data.reset_index(drop=True), validation_data.reset_index(drop=True)
