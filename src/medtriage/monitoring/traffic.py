"""Generate HTTP traffic for the MedTriage monitoring stack."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REQUESTS = 100
DEFAULT_DELAY_SECONDS = 0.05
DEFAULT_INVALID_RATIO = 0.1
HTTP_SUCCESS_MIN = 200
HTTP_CLIENT_ERROR_MIN = 400
HTTP_SERVER_ERROR_MIN = 500
HEALTH_REQUEST_THRESHOLD = 0.3

PREDICTION_TEXTS = (
    "Patient with severe chest pain and shortness of breath.",
    "Patient reports mild headache and no other relevant symptoms.",
    "Patient with persistent abdominal pain and nausea.",
    "Patient presents with fever and respiratory discomfort.",
    "Patient reports routine follow-up without acute complaints.",
)


@dataclass
class TrafficSummary:
    """Store the result of generated traffic."""

    total: int = 0
    successful: int = 0
    client_errors: int = 0
    server_errors: int = 0
    unexpected_failures: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate traffic for the MedTriage API."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=f"Number of requests to generate. Default: {DEFAULT_REQUESTS}",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=(f"Delay in seconds between requests. Default: {DEFAULT_DELAY_SECONDS}"),
    )
    parser.add_argument(
        "--invalid-ratio",
        type=float,
        default=DEFAULT_INVALID_RATIO,
        help=(
            "Fraction of requests that should intentionally fail validation. "
            f"Default: {DEFAULT_INVALID_RATIO}"
        ),
    )
    return parser.parse_args()


def send_request(
    request: urllib.request.Request,
) -> int:
    """Send an HTTP request and return the resulting status code."""
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def build_health_request(base_url: str) -> urllib.request.Request:
    """Build a health-check request."""
    return urllib.request.Request(
        url=f"{base_url}/health",
        method="GET",
    )


def build_prediction_request(
    base_url: str,
    *,
    valid: bool,
) -> urllib.request.Request:
    """Build a valid or intentionally invalid prediction request."""
    text = random.choice(PREDICTION_TEXTS) if valid else ""

    payload = json.dumps({"text": text}).encode("utf-8")

    return urllib.request.Request(
        url=f"{base_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def update_summary(summary: TrafficSummary, status_code: int) -> None:
    """Update traffic counters from an HTTP status code."""
    summary.total += 1

    if HTTP_SUCCESS_MIN <= status_code < HTTP_CLIENT_ERROR_MIN:
        summary.successful += 1
    elif HTTP_CLIENT_ERROR_MIN <= status_code < HTTP_SERVER_ERROR_MIN:
        summary.client_errors += 1
    elif status_code >= HTTP_SERVER_ERROR_MIN:
        summary.server_errors += 1


def generate_traffic(
    *,
    base_url: str,
    request_count: int,
    delay_seconds: float,
    invalid_ratio: float,
) -> TrafficSummary:
    """Generate a mix of health, prediction, and invalid requests."""
    summary = TrafficSummary()

    for _ in range(request_count):
        random_value = random.random()

        if random_value < invalid_ratio:
            request = build_prediction_request(
                base_url,
                valid=False,
            )
        elif random_value < HEALTH_REQUEST_THRESHOLD:
            request = build_health_request(base_url)
        else:
            request = build_prediction_request(
                base_url,
                valid=True,
            )

        try:
            status_code = send_request(request)
            update_summary(summary, status_code)
        except (urllib.error.URLError, TimeoutError):
            summary.total += 1
            summary.unexpected_failures += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return summary


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.requests <= 0:
        raise ValueError("--requests must be greater than zero.")

    if args.delay < 0:
        raise ValueError("--delay must be zero or greater.")

    if not 0 <= args.invalid_ratio <= 1:
        raise ValueError("--invalid-ratio must be between 0 and 1.")


def main() -> None:
    """Generate traffic and print a short execution summary."""
    args = parse_args()
    validate_arguments(args)

    summary = generate_traffic(
        base_url=args.base_url.rstrip("/"),
        request_count=args.requests,
        delay_seconds=args.delay,
        invalid_ratio=args.invalid_ratio,
    )

    print("\nMedTriage traffic generation complete")
    print("=" * 40)
    print(f"Total requests      : {summary.total}")
    print(f"Successful          : {summary.successful}")
    print(f"Client errors       : {summary.client_errors}")
    print(f"Server errors       : {summary.server_errors}")
    print(f"Unexpected failures : {summary.unexpected_failures}")


if __name__ == "__main__":
    main()
