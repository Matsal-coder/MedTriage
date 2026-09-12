"""Tests for sklearn to ONNX model export."""

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest

from medtriage.ci.prepare_model import create_ci_model
from medtriage.modeling.onnx_export import (
    export_model_to_onnx,
    load_baseline_pipeline,
)


def test_load_baseline_pipeline_missing_artifact(
    tmp_path: Path,
) -> None:
    """Fail clearly when the baseline artifact does not exist."""
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Baseline model artifact not found",
    ):
        load_baseline_pipeline(missing_path)


def test_export_model_to_onnx(
    tmp_path: Path,
) -> None:
    """Export a valid sklearn pipeline to a loadable ONNX model."""
    baseline_path = tmp_path / "baseline.joblib"
    onnx_path = tmp_path / "optimized.onnx"

    create_ci_model(baseline_path)

    exported_path = export_model_to_onnx(
        model_path=baseline_path,
        artifact_path=onnx_path,
    )

    assert exported_path == onnx_path
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    model = onnx.load(exported_path)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(
        str(exported_path),
        providers=["CPUExecutionProvider"],
    )

    assert session.get_inputs()
    assert session.get_outputs()
