"""Prometheus metrics for MedTriage HTTP observability."""

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS_TOTAL = Counter(
    "medtriage_http_requests_total",
    "Total number of HTTP requests received by the MedTriage API.",
    labelnames=("method", "route", "status_code"),
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "medtriage_http_request_duration_seconds",
    "End-to-end HTTP request duration in seconds.",
    labelnames=("method", "route"),
)

HTTP_ERRORS_TOTAL = Counter(
    "medtriage_http_errors_total",
    "Total number of HTTP requests completed with status code >= 400.",
    labelnames=("method", "route", "status_code"),
)

EXCLUDED_METRICS_PATHS = frozenset(
    {
        "/metrics",
        "/docs",
        "/openapi.json",
    }
)

UNMATCHED_ROUTE_LABEL = "__unmatched__"

HTTP_ERROR_STATUS_CODE = 400


def get_route_label(request: Request) -> str:
    """Return the normalized FastAPI route used as a Prometheus label."""
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)

    if isinstance(route_path, str):
        return route_path

    return UNMATCHED_ROUTE_LABEL


def record_http_request(
    *,
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record request count, duration, and errors."""
    status_code_label = str(status_code)

    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        route=route,
        status_code=status_code_label,
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method,
        route=route,
    ).observe(duration_seconds)

    if status_code >= HTTP_ERROR_STATUS_CODE:
        HTTP_ERRORS_TOTAL.labels(
            method=method,
            route=route,
            status_code=status_code_label,
        ).inc()


async def prometheus_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Measure HTTP requests while excluding observability-only routes."""
    if request.url.path in EXCLUDED_METRICS_PATHS:
        return await call_next(request)

    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        record_http_request(
            method=request.method,
            route=get_route_label(request),
            status_code=status_code,
            duration_seconds=perf_counter() - started_at,
        )


def metrics_response() -> Response:
    """Return the current Prometheus registry in exposition format."""
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )
