"""Prediction service backed by ONNX Runtime."""

from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from medtriage.config import (
    OPTIMIZED_MODEL_ARTIFACT_PATH,
    TRIAGE_CLASSES,
)


class OnnxPredictionService:
    """Load and serve predictions from the optimized ONNX model."""

    def __init__(
        self,
        model_path: Path = OPTIMIZED_MODEL_ARTIFACT_PATH,
    ) -> None:
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None

    def load(self) -> None:
        """Load the persisted ONNX model into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model artifact not found: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Predict the academic triage class using ONNX Runtime."""
        if self.session is None:
            raise RuntimeError("ONNX prediction model has not been loaded.")

        input_name = self.session.get_inputs()[0].name

        inputs = np.array(
            [[text]],
            dtype=object,
        )

        start = perf_counter()

        labels, probabilities = self.session.run(
            ["label", "probabilities"],
            {
                input_name: inputs,
            },
        )

        elapsed_ms = (perf_counter() - start) * 1000

        prediction = str(labels[0])

        probability_map = {
            label: float(probability)
            for label, probability in zip(
                TRIAGE_CLASSES,
                probabilities[0],
                strict=True,
            )
        }

        return prediction, probability_map, elapsed_ms
