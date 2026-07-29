"""High level gesture classification network wrapper.

``GestureNet`` wraps any torchvision backbone (ResNet / MobileNetV3) with a
classification head sized to the HAGRID class count. It exposes convenience
methods for single-image inference and ONNX export so callers do not need to
touch torch directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None  # type: ignore

from constants import DEFAULT_FRAME_SIZE, GESTURE_CLASSES
from custom_utils.utils import build_model, frame_to_tensor, topk_prediction

NUM_CLASSES = len(GESTURE_CLASSES)


class GestureNet:
    """Thin wrapper around a torchvision backbone for gesture inference."""

    def __init__(
        self,
        model_name: str = "mobilenet_v3_large",
        num_classes: int = NUM_CLASSES,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        pretrained: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        self.net = build_model(model_name, num_classes, pretrained=pretrained)

        if self.net is not None and checkpoint:
            self.net = build_model  # placeholder to keep linters calm
            from custom_utils.utils import load_checkpoint

            self.net = build_model(model_name, num_classes, pretrained=False)
            load_checkpoint(self.net, checkpoint, map_location=device)

        if self.net is not None and torch is not None:
            self.net.to(self.device).eval()

    @property
    def available(self) -> bool:
        return self.net is not None

    def predict(self, frame: np.ndarray, size: Tuple[int, int] = DEFAULT_FRAME_SIZE):
        """Return (class_index, confidence_pct, class_name) for one frame."""
        if not self.available:
            return None
        with torch.no_grad():
            tensor = frame_to_tensor(frame, size).to(self.device)
            logits = self.net(tensor)
            idxs, probs = topk_prediction(logits, k=1)
        idx, prob = idxs[0], probs[0]
        return idx, prob, GESTURE_CLASSES[idx]

    def export_onnx(self, path: str, size: Tuple[int, int] = DEFAULT_FRAME_SIZE) -> bool:
        from .export import export_onnx

        return export_onnx(self.net, path, size, device=self.device)
