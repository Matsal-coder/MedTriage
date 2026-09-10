"""Tests for dataset validation."""

import pandas as pd
import pytest

from medtriage.data.validation import validate_dataset


def test_valid_dataset_passes_validation() -> None:
    """A dataset respecting the expected contract should be accepted."""
    data = pd.DataFrame(
        {
            "medical_abstract": ["Example medical text.", "Another medical text."],
            "condition_label": [1, 5],
        }
    )

    validate_dataset(data)


def test_empty_dataset_is_rejected() -> None:
    """An empty dataset should be rejected."""
    data = pd.DataFrame(columns=["medical_abstract", "condition_label"])

    with pytest.raises(ValueError, match="must not be empty"):
        validate_dataset(data)


def test_missing_required_column_is_rejected() -> None:
    """A dataset without required columns should be rejected."""
    data = pd.DataFrame({"medical_abstract": ["Example medical text."]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(data)


def test_null_text_is_rejected() -> None:
    """Null medical abstracts should be rejected."""
    data = pd.DataFrame(
        {
            "medical_abstract": [None],
            "condition_label": [1],
        }
    )

    with pytest.raises(ValueError, match="null medical abstracts"):
        validate_dataset(data)


def test_empty_text_is_rejected() -> None:
    """Blank medical abstracts should be rejected."""
    data = pd.DataFrame(
        {
            "medical_abstract": ["   "],
            "condition_label": [1],
        }
    )

    with pytest.raises(ValueError, match="empty medical abstracts"):
        validate_dataset(data)


def test_invalid_condition_label_is_rejected() -> None:
    """Labels outside the corpus contract should be rejected."""
    data = pd.DataFrame(
        {
            "medical_abstract": ["Example medical text."],
            "condition_label": [99],
        }
    )

    with pytest.raises(ValueError, match="invalid condition labels"):
        validate_dataset(data)
