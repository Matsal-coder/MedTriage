"""API request and response schemas."""

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str


class PredictionRequest(BaseModel):
    """Request payload for medical text classification."""

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject whitespace-only medical texts."""
        normalized = value.strip()

        if not normalized:
            raise ValueError("Text must not be empty.")

        return normalized


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    prediction: str
    probabilities: dict[str, float]
    inference_time_ms: float
