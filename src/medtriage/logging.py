"""Logging configuration utilities."""

import logging

from medtriage.config import DEFAULT_LOG_LEVEL


def configure_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
