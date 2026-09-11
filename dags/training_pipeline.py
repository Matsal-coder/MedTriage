"""Airflow DAG for MedTriage model training and evaluation."""

from datetime import datetime

from airflow.sdk import dag, task

from medtriage.config import (
    EVALUATION_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from medtriage.data.loader import load_dataset
from medtriage.modeling.evaluate import run_evaluation
from medtriage.modeling.train import run_training


@dag(
    dag_id="medtriage_training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["medtriage", "training"],
)
def medtriage_training_pipeline():
    """Orchestrate MedTriage model training and evaluation."""

    @task
    def validate_data() -> None:
        """Validate the datasets required by training and evaluation."""
        load_dataset(TRAIN_DATA_PATH)
        load_dataset(TEST_DATA_PATH)

    @task
    def train_model() -> None:
        """Run the existing MedTriage training workflow."""
        run_training()

    @task
    def evaluate_model() -> None:
        """Run the existing MedTriage evaluation workflow."""
        run_evaluation()

    @task
    def validate_artifacts() -> None:
        """Validate the artifacts produced by the pipeline."""
        expected_artifacts = (
            MODEL_ARTIFACT_PATH,
            EVALUATION_ARTIFACT_PATH,
        )

        missing_artifacts = [
            artifact for artifact in expected_artifacts if not artifact.exists()
        ]

        if missing_artifacts:
            missing = ", ".join(str(path) for path in missing_artifacts)
            raise FileNotFoundError(
                f"Expected pipeline artifacts were not generated: {missing}"
            )

    data_validation = validate_data()
    training = train_model()
    evaluation = evaluate_model()
    artifact_validation = validate_artifacts()

    data_validation >> training >> evaluation >> artifact_validation


medtriage_training_pipeline()
