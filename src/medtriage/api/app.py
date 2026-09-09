"""FastAPI application entrypoint."""

from fastapi import FastAPI

from medtriage.api.schemas import HealthResponse
from medtriage.config import APP_NAME, APP_VERSION
from medtriage.logging import configure_logging

configure_logging()

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the current application health status."""
    return HealthResponse(status="ok")
