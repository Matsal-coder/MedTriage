"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from medtriage.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from medtriage.config import APP_NAME, APP_VERSION
from medtriage.logging import configure_logging
from medtriage.modeling.predict import PredictionService

configure_logging()


def create_prediction_service() -> PredictionService:
    """Create the prediction service used by the application."""
    return PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load application resources once during startup."""
    prediction_service = create_prediction_service()
    prediction_service.load()

    app.state.prediction_service = prediction_service

    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the current application health status."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Classify medical text using the loaded baseline model."""
    prediction_service: PredictionService = request.app.state.prediction_service

    prediction, probabilities, inference_time_ms = prediction_service.predict(
        payload.text
    )

    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        inference_time_ms=inference_time_ms,
    )
