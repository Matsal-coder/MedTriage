"""Baseline NLP model training utilities."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from medtriage.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    MODEL_ARTIFACT_PATH,
    RANDOM_SEED,
    TEXT_COLUMN,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TRAIN_DATA_PATH,
    TRIAGE_TARGET_COLUMN,
)
from medtriage.data.loader import (
    add_triage_labels,
    load_dataset,
    split_training_data,
)


def build_model_pipeline() -> Pipeline:
    """Build the baseline TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=TFIDF_MIN_DF,
                    max_df=TFIDF_MAX_DF,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_model(
    train_data: pd.DataFrame,
) -> Pipeline:
    """Train the baseline model using prepared training data."""
    model = build_model_pipeline()

    model.fit(
        train_data[TEXT_COLUMN],
        train_data[TRIAGE_TARGET_COLUMN],
    )

    return model


def persist_model(
    model: Pipeline,
    artifact_path: Path = MODEL_ARTIFACT_PATH,
) -> Path:
    """Persist the trained model pipeline to disk."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    return artifact_path


def run_training(
    dataset_path: Path = TRAIN_DATA_PATH,
    artifact_path: Path = MODEL_ARTIFACT_PATH,
) -> Path:
    """Run the complete baseline training workflow."""
    raw_data = load_dataset(dataset_path)
    prepared_data = add_triage_labels(raw_data)

    train_data, _ = split_training_data(prepared_data)

    model = train_model(train_data)

    return persist_model(model, artifact_path)


def main() -> None:
    """Run baseline model training from the command line."""
    artifact_path = run_training()
    print(f"Model artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
