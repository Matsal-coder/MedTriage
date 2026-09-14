"""Dataset quality diagnostics for MedTriage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from medtriage.config import (
    ORIGINAL_TARGET_COLUMN,
    TEST_DATA_PATH,
    TEXT_COLUMN,
    TRAIN_DATA_PATH,
    TRIAGE_TARGET_COLUMN,
)
from medtriage.data.loader import (
    add_triage_labels,
    load_dataset,
    split_training_data,
)

DEFAULT_DIAGNOSTIC_PATH = Path("artifacts/evaluation/data_overlap_analysis.json")


def calculate_split_statistics(
    data: pd.DataFrame,
) -> dict[str, int]:
    """Calculate basic duplication statistics for a dataset split."""
    duplicated_mask = data.duplicated(
        subset=[TEXT_COLUMN],
        keep=False,
    )

    return {
        "rows": len(data),
        "unique_texts": data[TEXT_COLUMN].nunique(),
        "rows_in_duplicated_groups": int(duplicated_mask.sum()),
        "duplicates_after_first": int(
            data.duplicated(
                subset=[TEXT_COLUMN],
                keep="first",
            ).sum()
        ),
    }


def calculate_overlap_statistics(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, int]:
    """Calculate text overlap from a reference split into a candidate split."""
    reference_texts = set(reference[TEXT_COLUMN])

    overlap_mask = candidate[TEXT_COLUMN].isin(reference_texts)
    overlapping_rows = candidate.loc[overlap_mask]

    shared_unique_texts = set(reference[TEXT_COLUMN]).intersection(
        set(candidate[TEXT_COLUMN])
    )

    return {
        "candidate_rows_with_text_in_reference": len(overlapping_rows),
        "shared_unique_texts": len(shared_unique_texts),
    }


def calculate_multilabel_statistics(
    data: pd.DataFrame,
    label_column: str,
) -> dict[str, Any]:
    """Find identical texts associated with more than one label."""
    label_counts = data.groupby(TEXT_COLUMN)[label_column].nunique()

    conflicting_texts = label_counts[label_counts > 1]

    conflicting_rows = data[data[TEXT_COLUMN].isin(conflicting_texts.index)]

    distribution = conflicting_texts.value_counts().sort_index().to_dict()

    return {
        "texts_with_multiple_labels": len(conflicting_texts),
        "rows_involved": len(conflicting_rows),
        "label_count_distribution": {
            str(label_count): int(text_count)
            for label_count, text_count in distribution.items()
        },
    }


def build_overlap_diagnostic(
    official_train: pd.DataFrame,
    effective_train: pd.DataFrame,
    validation: pd.DataFrame,
    official_test: pd.DataFrame,
) -> dict[str, Any]:
    """Build the complete reproducible data-quality diagnostic."""
    full_corpus = pd.concat(
        [
            official_train.assign(source="official_train"),
            official_test.assign(source="official_test"),
        ],
        ignore_index=True,
    )

    return {
        "scope": "medtriage_dataset_overlap_analysis",
        "methodology": {
            "text_column": TEXT_COLUMN,
            "original_target_column": ORIGINAL_TARGET_COLUMN,
            "triage_target_column": TRIAGE_TARGET_COLUMN,
            "split_source": ("existing split_training_data implementation"),
            "baseline_results_preserved": True,
        },
        "splits": {
            "official_train": calculate_split_statistics(official_train),
            "effective_train": calculate_split_statistics(effective_train),
            "validation": calculate_split_statistics(validation),
            "official_test": calculate_split_statistics(official_test),
            "full_corpus": calculate_split_statistics(full_corpus),
        },
        "overlap": {
            "effective_train_to_validation": (
                calculate_overlap_statistics(
                    effective_train,
                    validation,
                )
            ),
            "effective_train_to_official_test": (
                calculate_overlap_statistics(
                    effective_train,
                    official_test,
                )
            ),
            "validation_to_official_test": (
                calculate_overlap_statistics(
                    validation,
                    official_test,
                )
            ),
        },
        "multiple_labels": {
            "official_train_condition_label": (
                calculate_multilabel_statistics(
                    official_train,
                    ORIGINAL_TARGET_COLUMN,
                )
            ),
            "official_train_triage_label": (
                calculate_multilabel_statistics(
                    official_train,
                    TRIAGE_TARGET_COLUMN,
                )
            ),
            "full_corpus_condition_label": (
                calculate_multilabel_statistics(
                    full_corpus,
                    ORIGINAL_TARGET_COLUMN,
                )
            ),
            "full_corpus_triage_label": (
                calculate_multilabel_statistics(
                    full_corpus,
                    TRIAGE_TARGET_COLUMN,
                )
            ),
        },
    }


def persist_overlap_diagnostic(
    diagnostic: dict[str, Any],
    artifact_path: Path = DEFAULT_DIAGNOSTIC_PATH,
) -> Path:
    """Persist a dataset overlap diagnostic as JSON."""
    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with artifact_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            diagnostic,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return artifact_path


def run_overlap_diagnostic(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    artifact_path: Path = DEFAULT_DIAGNOSTIC_PATH,
) -> Path:
    """Run the complete dataset overlap analysis."""
    official_train_raw = load_dataset(train_path)
    official_test_raw = load_dataset(test_path)

    official_train = add_triage_labels(official_train_raw)
    official_test = add_triage_labels(official_test_raw)

    effective_train, validation = split_training_data(official_train)

    diagnostic = build_overlap_diagnostic(
        official_train=official_train,
        effective_train=effective_train,
        validation=validation,
        official_test=official_test,
    )

    return persist_overlap_diagnostic(
        diagnostic,
        artifact_path,
    )


def main() -> None:
    """Run the dataset quality diagnostic from the command line."""
    artifact_path = run_overlap_diagnostic()

    print(f"Data overlap diagnostic saved to: {artifact_path}")


if __name__ == "__main__":
    main()
