"""Model building and image conversion helpers.

Contains factory functions that instantiate torchvision backbones adapted to
the HAGRID gesture classes, plus utilities for converting between OpenCV
frames (BGR uint8) and torch tensors (RGB float normalised).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:  # Heavy import, guarded so the module loads without torch installed.
    import torch
    import torch.nn as nn
    from torchvision import models
except Exception:  # pragma: no cover - sandbox without torch
    torch = None
    nn = None
    models = None

from constants import GESTURE_CLASSES, DEFAULT_FRAME_SIZE

NUM_CLASSES = len(GESTURE_CLASSES)


MODEL_NAME_ALIASES = {
    "MobileNetV3_large": "mobilenet_v3_large",
    "MobileNetV3_small": "mobilenet_v3_small",
    "ResNet18": "resnet18",
    "ResNet34": "resnet34",
    "ResNet50": "resnet50",
    "ResNet152": "resnet152",
    "ConvNeXt_base": "convnext_base",
}


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_model_args(model_config: Any, num_classes: int, pretrained: bool) -> tuple[str, int, bool]:
    """Accept either a model name or the project's full OmegaConf config."""
    if isinstance(model_config, str):
        return MODEL_NAME_ALIASES.get(model_config, model_config), num_classes, pretrained

    config_model = _get_config_value(model_config, "model")
    if config_model is not None:
        model_name = _get_config_value(config_model, "name", "mobilenet_v3_large")
        pretrained = _get_config_value(config_model, "pretrained", pretrained)

        dataset_config = _get_config_value(model_config, "dataset")
        targets = _get_config_value(dataset_config, "targets") if dataset_config is not None else None
        if targets:
            num_classes = len(targets)

        return MODEL_NAME_ALIASES.get(model_name, model_name), num_classes, pretrained

    return MODEL_NAME_ALIASES.get(str(model_config), str(model_config)), num_classes, pretrained


def build_model(model_name: Any = "mobilenet_v3_large", num_classes: int = NUM_CLASSES, pretrained: bool = False):
    """Build a classification backbone with a fresh classifier head.

    Supported names: ``mobilenet_v3_large``, ``mobilenet_v3_small``,
    ``resnet18``, ``resnet34``, ``resnet50``. Falls back gracefully when
    torchvision is unavailable (returns ``None``).
    """
    if models is None:
        return None

    model_name, num_classes, pretrained = _resolve_model_args(model_name, num_classes, pretrained)
    weights_arg = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=weights_arg)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=weights_arg)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name in {"resnet18", "resnet34", "resnet50", "resnet152"}:
        net = getattr(models, model_name)(weights=weights_arg)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_base":
        net = models.convnext_base(weights=weights_arg)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return net


def frame_to_tensor(frame: np.ndarray, size: Tuple[int, int] = DEFAULT_FRAME_SIZE):
    """Convert a BGR uint8 frame to a normalised RGB float tensor (1, 3, H, W)."""
    if torch is None:
        return None
    import cv2

    frame = cv2.resize(frame, (size[1], size[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = frame.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def tensor_to_frame(tensor) -> np.ndarray:
    """Convert a (1, 3, H, W) float tensor back to a BGR uint8 frame."""
    import cv2

    arr = tensor.squeeze(0).detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def topk_prediction(logits, k: int = 1):
    """Return (indices, probabilities) for the top-k logits."""
    if torch is None:
        return [], []
    probs = torch.softmax(logits, dim=1)
    topk_probs, topk_idx = torch.topk(probs, k=k, dim=1)
    return topk_idx[0].tolist(), (topk_probs[0] * 100.0).tolist()


def load_checkpoint(model, ckpt_path: str, map_location: str = "cpu"):
    """Load model weights from a checkpoint file."""
    if torch is None:
        return model
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model
