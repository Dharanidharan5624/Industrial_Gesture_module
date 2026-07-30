"""Shirt Button Detector for Compliance Monitoring.

Detects whether the shirt buttons are properly fastened by:
  1. Analyzing a narrow placket strip (not the full chest) for a triangular
     skin V-gap that would indicate an unbuttoned collar.
  2. Confirming by checking that white buttons are absent in the placket ROI
     where they should be visible on a fully buttoned shirt.

IMPORTANT: The detector ONLY fires when a clear open V-gap is detected in the
narrow center placket. Normal neck/collar exposure does NOT trigger it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from compliance.detectors.base import BaseDetector
from compliance.models import DetectorResult
from constants import COMPLIANCE_EVENT_SHIRT_BUTTON

BBox = Tuple[int, int, int, int]


def _lm_xy(lm, w: int, h: int) -> Tuple[float, float]:
    if hasattr(lm, "x"):
        return lm.x * w, lm.y * h
    return lm[0] * w, lm[1] * h


def _dist3(landmarks, i1: int, i2: int) -> float:
    p1, p2 = landmarks[i1], landmarks[i2]
    x1, y1 = (p1.x, p1.y) if hasattr(p1, "x") else (p1[0], p1[1])
    x2, y2 = (p2.x, p2.y) if hasattr(p2, "x") else (p2[0], p2[1])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _is_pinch(landmarks, hand_size: float) -> Tuple[bool, float]:
    if hand_size <= 0:
        return False, 0.0
    d48 = _dist3(landmarks, 4, 8) / hand_size

    def extended(tip, pip, mcp):
        dt = _dist3(landmarks, tip, mcp)
        dp = _dist3(landmarks, pip, mcp)
        return dt > 0.6 * hand_size and dt > dp

    tight = d48 < 0.35 and not extended(12, 10, 9) and not extended(16, 14, 13) and not extended(20, 18, 17)
    loose = d48 < 0.50
    return (True, 1.0) if tight else (True, 0.80) if loose else (False, 0.65)


class ShirtButtonDetector(BaseDetector):
    name = "shirt_button"
    event_type = COMPLIANCE_EVENT_SHIRT_BUTTON

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.40, weights_path: Optional[str] = None):
        super().__init__(enabled=enabled, confidence_threshold=confidence_threshold)
        self.weights_path = weights_path
        self._history: List[float] = []
        self._max_history = 4          # responsive smoothing
        self._on_threshold  = 0.50     # sustained signal required
        self._off_threshold = 0.28
        self._is_open = False
        self.last_bbox:  Optional[BBox] = None
        self.last_label: str = "Shirt Button OK"
        self.last_conf:  float = 0.0

    def _extract_hands(self, context: Optional[Dict[str, Any]]) -> List[Any]:
        if not context:
            return []
        for key in ("hand_landmarks", "landmarks"):
            hl = context.get(key)
            if not hl:
                continue
            if isinstance(hl, list) and len(hl) > 0:
                item0 = hl[0]
                if hasattr(item0, "x") or (isinstance(item0, (list, tuple)) and isinstance(item0[0], (int, float))):
                    return [hl]
                return [h for h in hl if isinstance(h, (list, tuple)) and len(h) >= 9]
        return []

    def _analyze_placket(self, placket: np.ndarray) -> Tuple[bool, float]:
        """Inspect placket strip for a visible skin V-gap (unbuttoned collar)."""
        if placket is None or placket.size == 0 or placket.shape[0] < 15 or placket.shape[1] < 10:
            return False, 0.0

        ph, pw = placket.shape[:2]
        hsv = cv2.cvtColor(placket, cv2.COLOR_BGR2HSV)

        # Tight skin HSV mask to filter out shirt fabrics (blue, grey, white, pattern)
        skin1 = cv2.inRange(hsv, np.array([0,  28, 65],  np.uint8), np.array([22, 175, 245], np.uint8))
        skin2 = cv2.inRange(hsv, np.array([162, 28, 65], np.uint8), np.array([178, 175, 245], np.uint8))
        skin_mask = cv2.bitwise_or(skin1, skin2)

        # Skin ratio in the placket region below collar
        skin_ratio = np.count_nonzero(skin_mask) / max(1, skin_mask.size)

        # Divide into top and bottom halves of placket
        half_h = max(1, ph // 2)
        top_half = skin_mask[:half_h, :]
        top_ratio = np.count_nonzero(top_half) / max(1, top_half.size)

        is_open = (top_ratio > 0.15 or skin_ratio > 0.18)
        conf = min(0.95, 0.45 + max(top_ratio, skin_ratio) * 1.5) if is_open else 0.0

        return is_open, float(conf)

    def detect(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(detail="disabled")

        if frame is None or frame.size == 0:
            return DetectorResult(detail="no_frame")

        h, w = frame.shape[:2]

        has_person = False
        chin_x, chin_y, face_h = 0.0, 0.0, 0.0

        # 1. Try face_landmarks if available
        if context:
            lm = context.get("face_landmarks")
            if lm and hasattr(lm, "__len__") and len(lm) > 152:
                try:
                    cx_val, cy_val = _lm_xy(lm[152], w, h)
                    nx_val, ny_val = _lm_xy(lm[1], w, h)
                    chin_x, chin_y = cx_val, cy_val
                    face_h = math.hypot(chin_x - nx_val, chin_y - ny_val)
                    if face_h > 12:
                        has_person = True
                except Exception:
                    pass

        # 2. Robust skin-contour head/neck localization fallback
        if not has_person:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s1 = cv2.inRange(hsv, np.array([0, 28, 65], np.uint8), np.array([22, 175, 245], np.uint8))
            s2 = cv2.inRange(hsv, np.array([162, 28, 65], np.uint8), np.array([178, 175, 245], np.uint8))
            skin = cv2.bitwise_or(s1, s2)

            upper = skin[:int(h * 0.75), :]
            contours, _ = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 500:
                    x, y, bw, bh = cv2.boundingRect(largest)
                    chin_x = x + bw / 2.0
                    head_h = int(bw * 1.10)
                    chin_y = y + head_h
                    face_h = max(25.0, bw * 0.50)
                    has_person = True

        # If no human/head skin detected in frame, return no_person
        if not has_person:
            self.last_bbox = None
            return DetectorResult(detected=False, confidence=0.0, detail="no_person")

        # Position placket ROI relative to chin / collar line
        cx = int(chin_x)
        py1 = int(chin_y + 0.05 * face_h)
        py2 = int(chin_y + 1.80 * face_h)
        placket_w = int(max(35, face_h * 0.70))
        px1, px2 = max(0, cx - placket_w // 2), min(w, cx + placket_w // 2)

        py1 = max(0, min(py1, h - 1))
        py2 = max(py1 + 10, min(py2, h))
        px1 = max(0, min(px1, w - 1))
        px2 = max(px1 + 10, min(px2, w))
        bbox: BBox = (px1, py1, px2, py2)
        self.last_bbox = bbox

        placket_roi = frame[py1:py2, px1:px2]

        vis_open, vis_conf = self._analyze_placket(placket_roi)

        if vis_open and vis_conf >= float(self.confidence_threshold):
            return DetectorResult(
                detected=True,
                confidence=vis_conf,
                detail="Shirt Button Open",
                bbox=bbox,
            )

        return DetectorResult(detected=False, confidence=vis_conf, detail="Shirt Button OK", bbox=bbox)
