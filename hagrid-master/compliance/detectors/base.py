"""Base interface for pluggable compliance detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from compliance.models import DetectorResult


class BaseDetector(ABC):
    """Pluggable detector: subclasses implement ``detect`` only."""

    name: str = "base"
    event_type: str = ""
    model_unavailable_logged: bool = False

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.45):
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.weights_path: Optional[str] = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_confidence_threshold(self, value: float) -> None:
        self.confidence_threshold = float(value)

    @abstractmethod
    def detect(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        """Run detection on a BGR frame. Return a DetectorResult."""

    def unavailable_result(self, detail: str = "model_unavailable") -> DetectorResult:
        return DetectorResult(detected=False, confidence=0.0, detail=detail)
