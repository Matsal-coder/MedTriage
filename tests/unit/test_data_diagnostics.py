"""Tests for MedTriage dataset diagnostics."""

from __future__ import annotations

import json

import pandas as pd

from medtriage.config import (
    ORIGINAL_TARGET_COLUMN,
    TEXT_COLUMN,
    TRIAGE_TARGET_COLUMN,
)
from medtriage.data.diagnostics import (
    build_overlap_diagnostic,
    calculate_multilabel_statistics,
    calculate_overlap_statistics,
    calculate_split_statistics,
    persist_overlap_diagnostic,
)


def make_dataframe(
    texts: list[str],
    condition_labels: list[int],
    triage_labels: list[str],
) -> pd.DataFrame:
    """Build a minimal prepared MedTriage dataframe."""
    return pd.DataFrame(
        {
            TEXT_COLUMN: texts,
            ORIGINAL_TARGET_COLUMN: condition_labels,
            TRIAGE_TARGET_COLUMN: triage_labels,
        }
    )


def test_calculate_split_statistics() -> None:
    data = make_dataframe(
        texts=[
            "Repeated text",
            "Repeated text",
            "Unique text",
        ],
        condition_labels=[1, 4, 5],
        triage_labels=[
            "urgent",
            "urgent",
            "normal",
        ],
    )

    result = calculate_split_statistics(data)

    assert result == {
        "rows": 3,
        "unique_texts": 2,
        "rows_in_duplicated_groups": 2,
        "duplicates_after_first": 1,
    }


def test_calculate_overlap_statistics() -> None:
    reference = make_dataframe(
        texts=[
            "Shared text",
            "Reference only",
        ],
        condition_labels=[1, 5],
        triage_labels=[
            "urgent",
            "normal",
        ],
    )

    candidate = make_dataframe(
        texts=[
            "Shared text",
            "Shared text",
            "Candidate only",
        ],
        condition_labels=[1, 1, 2],
        triage_labels=[
            "urgent",
            "urgent",
            "attention",
        ],
    )

    result = calculate_overlap_statistics(
        reference,
        candidate,
    )

    assert result == {
        "candidate_rows_with_text_in_reference": 2,
        "shared_unique_texts": 1,
    }


def test_calculate_multilabel_statistics() -> None:
    data = make_dataframe(
        texts=[
            "Conflicting text",
            "Conflicting text",
            "Stable text",
        ],
        condition_labels=[1, 5, 2],
        triage_labels=[
            "urgent",
            "normal",
            "attention",
        ],
    )

    result = calculate_multilabel_statistics(
        data,
        ORIGINAL_TARGET_COLUMN,
    )

    assert result == {
        "texts_with_multiple_labels": 1,
        "rows_involved": 2,
        "label_count_distribution": {
            "2": 1,
        },
    }


def test_build_overlap_diagnostic() -> None:
    official_train = make_dataframe(
        texts=[
            "Shared validation",
            "Shared validation",
            "Shared test",
            "Training only",
        ],
        condition_labels=[1, 5, 2, 3],
        triage_labels=[
            "urgent",
            "normal",
            "attention",
            "attention",
        ],
    )

    effective_train = official_train.iloc[[0, 2, 3]].reset_index(drop=True)

    validation = official_train.iloc[[1]].reset_index(drop=True)

    official_test = make_dataframe(
        texts=[
            "Shared test",
            "Test only",
        ],
        condition_labels=[2, 4],
        triage_labels=[
            "attention",
            "urgent",
        ],
    )

    result = build_overlap_diagnostic(
        official_train=official_train,
        effective_train=effective_train,
        validation=validation,
        official_test=official_test,
    )

    assert (
        result["overlap"]["effective_train_to_validation"][
            "candidate_rows_with_text_in_reference"
        ]
        == 1
    )

    assert (
        result["overlap"]["effective_train_to_official_test"][
            "candidate_rows_with_text_in_reference"
        ]
        == 1
    )

    assert (
        result["multiple_labels"]["official_train_condition_label"][
            "texts_with_multiple_labels"
        ]
        == 1
    )


def test_persist_overlap_diagnostic(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "data_overlap_analysis.json"

    diagnostic = {
        "scope": "test",
        "rows": 10,
    }

    saved_path = persist_overlap_diagnostic(
        diagnostic,
        artifact_path,
    )

    assert saved_path == artifact_path
    assert artifact_path.exists()

    with artifact_path.open(encoding="utf-8") as file:
        persisted = json.load(file)

    assert persisted == diagnostic
