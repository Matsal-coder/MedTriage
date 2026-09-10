"""Tests for application configuration."""

from medtriage.config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_LOG_LEVEL,
    EVALUATION_ARTIFACT_FILENAME,
    LOGISTIC_REGRESSION_MAX_ITER,
    MODEL_ARTIFACT_FILENAME,
    ORIGINAL_TARGET_COLUMN,
    RANDOM_SEED,
    TEST_DATA_FILENAME,
    TEXT_COLUMN,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TRAIN_DATA_FILENAME,
    TRIAGE_CLASSES,
    TRIAGE_TARGET_COLUMN,
    VALIDATION_SIZE,
)

EXPECTED_RANDOM_SEED = 837
EXPECTED_VALIDATION_SIZE = 0.20
EXPECTED_MAX_ITER = 1000
EXPECTED_TFIDF_MIN_DF = 2
EXPECTED_TFIDF_MAX_DF = 0.95


def test_model_configuration() -> None:
    """Model configuration should expose centralized baseline parameters."""
    assert MODEL_ARTIFACT_FILENAME == "baseline_pipeline.joblib"
    assert TFIDF_NGRAM_RANGE == (1, 2)
    assert TFIDF_MIN_DF == EXPECTED_TFIDF_MIN_DF
    assert TFIDF_MAX_DF == EXPECTED_TFIDF_MAX_DF
    assert LOGISTIC_REGRESSION_MAX_ITER == EXPECTED_MAX_ITER
    assert EVALUATION_ARTIFACT_FILENAME == "evaluation.json"
    assert TRIAGE_CLASSES == ("attention", "normal", "urgent")


def test_application_configuration() -> None:
    """Application configuration should expose expected defaults."""
    assert APP_NAME == "MedTriage MLOps"
    assert APP_VERSION == "0.1.0"
    assert DEFAULT_LOG_LEVEL == "INFO"


def test_data_configuration() -> None:
    """Data configuration should expose centralized dataset contracts."""
    assert RANDOM_SEED == EXPECTED_RANDOM_SEED
    assert TRAIN_DATA_FILENAME == "medical_tc_train.csv"
    assert TEST_DATA_FILENAME == "medical_tc_test.csv"
    assert TEXT_COLUMN == "medical_abstract"
    assert ORIGINAL_TARGET_COLUMN == "condition_label"
    assert TRIAGE_TARGET_COLUMN == "triage_label"
    assert VALIDATION_SIZE == EXPECTED_VALIDATION_SIZE
