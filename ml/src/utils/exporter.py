from __future__ import annotations

from pathlib import Path
from typing import Optional

import tensorflow as tf


def export_tflite(model: tf.keras.Model, export_path: Path) -> Path:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    export_path.write_bytes(tflite_model)
    return export_path


def export_onnx(model: tf.keras.Model, export_path: Path, opset: int = 13) -> Optional[Path]:
    try:
        import tf2onnx
        from tf2onnx import tf_loader
    except ImportError as exc:  # pragma: no cover - opsiyonel
        raise RuntimeError("tf2onnx paketi yüklü değil.") from exc

    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset, output_path=str(export_path))
    if model_proto:
        return export_path
    return None

