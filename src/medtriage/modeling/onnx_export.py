"""ONNX export utilities for the trained baseline model."""

from pathlib import Path

import joblib
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
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


def convert_pipeline_to_onnx(
    model: Pipeline,
) -> onnx.ModelProto:
    """Convert the sklearn text classification pipeline to ONNX."""
    classifier = model.named_steps.get("classifier")

    if classifier is None:
        raise ValueError("Baseline pipeline must contain a 'classifier' step.")

    conversion_options = {
        id(classifier): {
            "zipmap": False,
        }
    }

    onnx_model = convert_sklearn(
        model,
        initial_types=[
            (
                "text",
                StringTensorType([None, 1]),
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
    """Validate that ONNX Runtime can load the exported model."""
    ort.InferenceSession(
        str(artifact_path),
        providers=["CPUExecutionProvider"],
    )


def export_model_to_onnx(
    model_path: Path = MODEL_ARTIFACT_PATH,
    artifact_path: Path = OPTIMIZED_MODEL_ARTIFACT_PATH,
) -> Path:
    """Convert and persist the baseline sklearn pipeline as ONNX."""
    model = load_baseline_pipeline(model_path)

    onnx_model = convert_pipeline_to_onnx(model)

    exported_path = persist_onnx_model(
        onnx_model,
        artifact_path,
    )

    validate_onnx_runtime(exported_path)

    return exported_path


def main() -> None:
    """Export the baseline model to ONNX from the command line."""
    artifact_path = export_model_to_onnx()

    print(f"ONNX model artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
