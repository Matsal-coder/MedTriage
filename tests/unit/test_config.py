"""Tests for application configuration."""

from medtriage.config import APP_NAME, APP_VERSION, DEFAULT_LOG_LEVEL


def test_application_configuration() -> None:
    """Application configuration should expose expected defaults."""
    assert APP_NAME == "MedTriage MLOps"
    assert APP_VERSION == "0.1.0"
    assert DEFAULT_LOG_LEVEL == "INFO"
