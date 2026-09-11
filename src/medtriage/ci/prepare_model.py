"""Prepare a minimal model artifact for Docker validation in CI."""

from pathlib import Path

import pandas as pd

from medtriage.config import (
    MODEL_ARTIFACT_PATH,
    TEXT_COLUMN,
    TRIAGE_TARGET_COLUMN,
)
from medtriage.modeling.train import persist_model, train_model

CI_TRAINING_ROWS = (
    {
        TEXT_COLUMN: "urgent emergency patient severe trauma",
        TRIAGE_TARGET_COLUMN: "urgent",
    },
    {
        TEXT_COLUMN: "urgent emergency patient critical injury",
        TRIAGE_TARGET_COLUMN: "urgent",
    },
    {
        TEXT_COLUMN: "attention patient moderate symptoms evaluation",
        TRIAGE_TARGET_COLUMN: "attention",
    },
    {
        TEXT_COLUMN: "attention patient followup clinical evaluation",
        TRIAGE_TARGET_COLUMN: "attention",
    },
    {
        TEXT_COLUMN: "normal patient stable routine checkup",
        TRIAGE_TARGET_COLUMN: "normal",
    },
    {
        TEXT_COLUMN: "normal patient stable routine screening",
        TRIAGE_TARGET_COLUMN: "normal",
    },
)


def create_ci_model(
    artifact_path: Path = MODEL_ARTIFACT_PATH,
) -> Path:
    """Create a minimal runtime-compatible model artifact for CI."""
    training_data = pd.DataFrame(CI_TRAINING_ROWS)

    model = train_model(training_data)

    return persist_model(model, artifact_path)


def main() -> None:
    """Create the temporary CI model artifact."""
    artifact_path = create_ci_model()
    print(f"CI model artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
