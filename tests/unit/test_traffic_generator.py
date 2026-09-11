"""Unit tests for the monitoring traffic generator."""

import argparse

import pytest

from medtriage.monitoring.traffic import (
    TrafficSummary,
    update_summary,
    validate_arguments,
)


def test_update_summary_counts_successful_request() -> None:
    """A successful HTTP response should increment the success counter."""
    summary = TrafficSummary()

    update_summary(summary, 200)

    assert summary.total == 1
    assert summary.successful == 1
    assert summary.client_errors == 0
    assert summary.server_errors == 0


def test_update_summary_counts_client_error() -> None:
    """A 4xx response should increment the client error counter."""
    summary = TrafficSummary()

    update_summary(summary, 422)

    assert summary.total == 1
    assert summary.successful == 0
    assert summary.client_errors == 1
    assert summary.server_errors == 0


def test_update_summary_counts_server_error() -> None:
    """A 5xx response should increment the server error counter."""
    summary = TrafficSummary()

    update_summary(summary, 500)

    assert summary.total == 1
    assert summary.successful == 0
    assert summary.client_errors == 0
    assert summary.server_errors == 1


def test_validate_arguments_accepts_valid_values() -> None:
    """Valid traffic generation arguments should be accepted."""
    args = argparse.Namespace(
        requests=10,
        delay=0.0,
        invalid_ratio=0.1,
    )

    validate_arguments(args)


@pytest.mark.parametrize(
    ("requests", "delay", "invalid_ratio", "message"),
    [
        (0, 0.0, 0.1, "--requests must be greater than zero."),
        (10, -0.1, 0.1, "--delay must be zero or greater."),
        (10, 0.0, -0.1, "--invalid-ratio must be between 0 and 1."),
        (10, 0.0, 1.1, "--invalid-ratio must be between 0 and 1."),
    ],
)
def test_validate_arguments_rejects_invalid_values(
    requests: int,
    delay: float,
    invalid_ratio: float,
    message: str,
) -> None:
    """Invalid traffic generation arguments should raise ValueError."""
    args = argparse.Namespace(
        requests=requests,
        delay=delay,
        invalid_ratio=invalid_ratio,
    )

    with pytest.raises(ValueError, match=message):
        validate_arguments(args)
