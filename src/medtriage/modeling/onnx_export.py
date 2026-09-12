"""ONNX export utilities for the trained baseline classifier."""

from pathlib import Path

import joblib
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from medtriage.config import (
    MODEL_ARTIFACT_PATH,
    OPTIMIZED_MODEL_ARTIFACT_PATH,
)


def load_baseline_pipeline(
    model_path: Path = MODEL_ARTIFACT_PATH,
) -> Pipeline:
    """Load the persisted sklearn baseline pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model artifact not found: {model_path}")

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Baseline model artifact must contain a sklearn Pipeline.")

    return model


def get_classifier(model: Pipeline) -> LogisticRegression:
    """Extract the logistic regression classifier from the baseline pipeline."""
    classifier = model.named_steps.get("classifier")

    if not isinstance(classifier, LogisticRegression):
        raise TypeError(
            "Baseline pipeline must contain a LogisticRegression 'classifier' step."
        )

    return classifier


def convert_classifier_to_onnx(
    classifier: LogisticRegression,
) -> onnx.ModelProto:
    """Convert the trained logistic regression classifier to ONNX."""
    conversion_options = {
        id(classifier): {
            "zipmap": False,
        }
    }

    onnx_model = convert_sklearn(
        classifier,
        initial_types=[
            (
                "features",
                FloatTensorType(
                    [
                        None,
                        classifier.n_features_in_,
                    ]
                ),
            )
        ],
        options=conversion_options,
    )

    onnx.checker.check_model(onnx_model)

    return onnx_model


def persist_onnx_model(
    model: onnx.ModelProto,
    artifact_path: Path = OPTIMIZED_MODEL_ARTIFACT_PATH,
) -> Path:
    """Persist a validated ONNX model to disk."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    onnx.save_model(model, artifact_path)

    return artifact_path


def validate_onnx_runtime(
    artifact_path: Path,
) -> None:
    """Validate that ONNX Runtime can load the exported classifier."""
    ort.InferenceSession(
        str(artifact_path),
        providers=["CPUExecutionProvider"],
    )


def export_model_to_onnx(
    model_path: Path = MODEL_ARTIFACT_PATH,
    artifact_path: Path = OPTIMIZED_MODEL_ARTIFACT_PATH,
) -> Path:
    """Export the baseline logistic regression classifier to ONNX."""
    model = load_baseline_pipeline(model_path)
    classifier = get_classifier(model)

    onnx_model = convert_classifier_to_onnx(classifier)

    exported_path = persist_onnx_model(
        onnx_model,
        artifact_path,
    )

    validate_onnx_runtime(exported_path)

    return exported_path


def main() -> None:
    """Export the trained baseline classifier to ONNX."""
    artifact_path = export_model_to_onnx()

    print(f"ONNX model artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
