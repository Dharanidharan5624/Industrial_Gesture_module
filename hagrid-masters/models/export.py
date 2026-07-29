"""ONNX export and runtime helpers.

Serialises a trained backbone into the ONNX graph format and provides a
``load_onnx_session`` helper that returns a high-speed onnxruntime inference
session. Both paths degrade gracefully when the optional onnx / onnxruntime
packages are missing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from constants import DEFAULT_FRAME_SIZE


def export_onnx(model, path: str, size: Tuple[int, int] = DEFAULT_FRAME_SIZE, device: str = "cpu") -> bool:
    """Export a torch model to ``path`` as an ONNX graph.

    Returns ``True`` on success, ``False`` when torch/onnx are unavailable.
    """
    if torch is None or model is None:
        return False
    dummy = torch.randn(1, 3, size[0], size[1], device=device)
    try:
        torch.onnx.export(
            model,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        return True
    except Exception as exc:  # pragma: no cover - export depends on torch
        print(f"[ONNX] export failed: {exc}")
        return False


def load_onnx_session(path: str, providers: Optional[list] = None):
    """Return an onnxruntime InferenceSession or ``None`` if unavailable."""
    if ort is None:
        return None
    providers = providers or ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(path, providers=providers)
    except Exception as exc:  # pragma: no cover - depends on runtime
        print(f"[ONNX] session load failed: {exc}")
        return None


def run_onnx(session, frame: np.ndarray, size: Tuple[int, int] = DEFAULT_FRAME_SIZE):
    """Pre-process a BGR frame and run a single ONNX inference.

    Returns the raw logits array.
    """
    import cv2

    frame = cv2.resize(frame, (size[1], size[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr = np.transpose(frame, (2, 0, 1))[None]
    inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, inputs)
    return outputs[0]
