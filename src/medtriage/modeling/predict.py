"""Prediction service for the trained baseline model."""

from pathlib import Path
from time import perf_counter

import joblib
from sklearn.pipeline import Pipeline

from medtriage.config import MODEL_ARTIFACT_PATH


class PredictionService:
    """Load and serve predictions from the persisted baseline model."""

    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_PATH,
    ) -> None:
        self.model_path = model_path
        self.model: Pipeline | None = None

    def load(self) -> None:
        """Load the persisted model into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Predict the academic triage class for a medical text."""
        if self.model is None:
            raise RuntimeError("Prediction model has not been loaded.")

        start = perf_counter()

        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]

        elapsed_ms = (perf_counter() - start) * 1000

        probability_map = {
            label: float(probability)
            for label, probability in zip(
                self.model.classes_,
                probabilities,
                strict=True,
            )
        }

        return str(prediction), probability_map, elapsed_ms
