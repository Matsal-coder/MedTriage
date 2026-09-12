"""Prediction service backed by sklearn preprocessing and ONNX Runtime."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from medtriage.config import (
    MODEL_ARTIFACT_PATH,
    OPTIMIZED_MODEL_ARTIFACT_PATH,
)


class OnnxPredictionService:
    """Serve predictions using sklearn TF-IDF and an ONNX classifier."""

    def __init__(
        self,
        model_path: Path = OPTIMIZED_MODEL_ARTIFACT_PATH,
        baseline_model_path: Path = MODEL_ARTIFACT_PATH,
    ) -> None:
        self.model_path = model_path
        self.baseline_model_path = baseline_model_path

        self.session: ort.InferenceSession | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.classes: tuple[str, ...] | None = None

    def load(self) -> None:
        """Load sklearn preprocessing and the ONNX classifier."""
        if not self.baseline_model_path.exists():
            raise FileNotFoundError(
                f"Baseline model artifact not found: {self.baseline_model_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model artifact not found: {self.model_path}")

        baseline_model = joblib.load(self.baseline_model_path)

        if not isinstance(baseline_model, Pipeline):
            raise TypeError("Baseline model artifact must contain a sklearn Pipeline.")

        vectorizer = baseline_model.named_steps.get("tfidf")
        classifier = baseline_model.named_steps.get("classifier")

        if not isinstance(vectorizer, TfidfVectorizer):
            raise TypeError(
                "Baseline pipeline must contain a TfidfVectorizer 'tfidf' step."
            )

        if classifier is None or not hasattr(classifier, "classes_"):
            raise TypeError("Baseline pipeline classifier must expose fitted classes.")

        self.vectorizer = vectorizer
        self.classes = tuple(str(label) for label in classifier.classes_)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(
        self,
        text: str,
    ) -> tuple[str, dict[str, float], float]:
        """Predict using sklearn TF-IDF followed by ONNX classification."""
        if self.session is None or self.vectorizer is None or self.classes is None:
            raise RuntimeError("ONNX prediction model has not been loaded.")

        start = perf_counter()

        features = self.vectorizer.transform([text])

        dense_features = features.astype(np.float32).toarray()

        input_name = self.session.get_inputs()[0].name

        labels, probabilities = self.session.run(
            ["label", "probabilities"],
            {
                input_name: dense_features,
            },
        )

        elapsed_ms = (perf_counter() - start) * 1000

        prediction = str(labels[0])

        probability_map = {
            label: float(probability)
            for label, probability in zip(
                self.classes,
                probabilities[0],
                strict=True,
            )
        }

        return prediction, probability_map, elapsed_ms
