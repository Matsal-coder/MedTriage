"""Tests for application configuration."""

from medtriage.config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_LOG_LEVEL,
    ORIGINAL_TARGET_COLUMN,
    RANDOM_SEED,
    TEST_DATA_FILENAME,
    TEXT_COLUMN,
    TRAIN_DATA_FILENAME,
    TRIAGE_TARGET_COLUMN,
    VALIDATION_SIZE,
)

EXPECTED_RANDOM_SEED = 837
EXPECTED_VALIDATION_SIZE = 0.20


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
