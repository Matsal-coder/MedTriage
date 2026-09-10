"""Application configuration."""

from pathlib import Path

APP_NAME = "MedTriage MLOps"
APP_VERSION = "0.1.0"
DEFAULT_LOG_LEVEL = "INFO"

RANDOM_SEED = 837

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DATA_FILENAME = "medical_tc_train.csv"
TEST_DATA_FILENAME = "medical_tc_test.csv"

TRAIN_DATA_PATH = RAW_DATA_DIR / TRAIN_DATA_FILENAME
TEST_DATA_PATH = RAW_DATA_DIR / TEST_DATA_FILENAME

TEXT_COLUMN = "medical_abstract"
ORIGINAL_TARGET_COLUMN = "condition_label"
TRIAGE_TARGET_COLUMN = "triage_label"

VALIDATION_SIZE = 0.20
