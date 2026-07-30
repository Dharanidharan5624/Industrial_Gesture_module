"""Writing Activity Detector for Compliance Monitoring.

Detects writing activity only when:
1. The hand is in a writing grip pose (thumb, index, and middle fingers clustered).
2. A pen, pencil, or similar writing instrument is visually detected in the hand
   (using Hough Lines and contour aspect ratio analysis in the finger-tip ROI).
3. Active writing motion (small-scale, high-frequency oscillatory motion) is performed.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from compliance.detectors.base import BaseDetector
from compliance.models import DetectorResult
from constants import COMPLIANCE_EVENT_WRITING


def _dist2d(p1, p2) -> float:
    x1, y1 = (p1.x, p1.y) if hasattr(p1, "x") else (p1[0], p1[1])
    x2, y2 = (p2.x, p2.y) if hasattr(p2, "x") else (p2[0], p2[1])
    return math.hypot(x1 - x2, y1 - y2)


class WritingDetector(BaseDetector):
    name = "writing"
    event_type = COMPLIANCE_EVENT_WRITING

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.45,
        weights_path: Optional[str] = None,
    ):
        super().__init__(enabled=enabled, confidence_threshold=confidence_threshold)
        self.weights_path = weights_path
        self._history_xy: List[Tuple[float, float]] = []
        self._max_history = 20
        self._last_state = "idle"
        self._smooth_score = 0.0

    def _verify_writing_grip(self, landmarks, w: int, h: int) -> Tuple[bool, float]:
        """Verify if hand landmarks match a writing or pointing grip pose.

        In a writing/pointing grip:
        - Thumb tip (4) and index tip (8) are close (pinching the pen).
        - Index finger is extended forward.
        - Middle finger may be nearby (supporting) or slightly extended.
        """
        if landmarks is None or len(landmarks) < 21:
            return False, 0.0

        hand_scale = _dist2d(landmarks[0], landmarks[9])
        if hand_scale <= 0:
            return False, 0.0

        d_thumb_index = _dist2d(landmarks[4], landmarks[8]) / hand_scale
        d_thumb_middle = _dist2d(landmarks[4], landmarks[12]) / hand_scale
        d_index_middle = _dist2d(landmarks[8], landmarks[12]) / hand_scale

        d_index_mcp_tip = _dist2d(landmarks[5], landmarks[8]) / hand_scale
        d_thumb_ip_tip = _dist2d(landmarks[3], landmarks[4]) / hand_scale

        grip_close = d_thumb_index < 0.45
        finger_extended = d_index_mcp_tip > 0.30

        is_clustered = grip_close and finger_extended
        confidence = 0.0
        if is_clustered:
            confidence = max(0.25, 1.0 - (d_thumb_index / 0.50))
            if d_thumb_middle < 0.60:
                confidence = min(1.0, confidence + 0.15)

        return is_clustered, confidence

    def _detect_writing_instrument(self, frame: np.ndarray, landmarks, w: int, h: int) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """Detect a pen, pencil, marker, or note/paper edge in hand ROI.

        Crops an ROI around finger tips and uses Hough lines + contour analysis.
        """
        if landmarks is None or len(landmarks) < 13:
            return False, 0.0, None

        try:
            pts = []
            for idx in [4, 8, 12]:
                lm = landmarks[idx]
                px = lm.x * w if hasattr(lm, "x") else lm[0] * w
                py = lm.y * h if hasattr(lm, "y") else lm[1] * h
                pts.append((px, py))
            cx = int(sum(p[0] for p in pts) / 3.0)
            cy = int(sum(p[1] for p in pts) / 3.0)
        except Exception:
            return False, 0.0, None

        roi_size = 70
        rx1, rx2 = max(0, cx - roi_size), min(w, cx + roi_size)
        ry1, ry2 = max(0, cy - roi_size), min(h, cy + roi_size)

        if (rx2 - rx1) < 16 or (ry2 - ry1) < 16:
            return False, 0.0, None

        roi = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi / 180,
            threshold=14,
            minLineLength=10,
            maxLineGap=8
        )

        has_line = False
        line_score = 0.0
        if lines is not None and len(lines) > 0:
            has_line = True
            line_score = min(0.90, 0.40 + len(lines) * 0.06)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_elongated_contour = False
        contour_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 8 <= area <= 1500:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    length = max(width, height)
                    if aspect_ratio >= 2.0 and length >= 10:
                        has_elongated_contour = True
                        contour_score = max(contour_score, min(0.90, 0.30 + aspect_ratio * 0.10))

        dark_mask = gray < 100
        dark_ratio = float(np.count_nonzero(dark_mask)) / max(1.0, float(gray.size))
        if 0.05 < dark_ratio < 0.60:
            lines_dark = cv2.HoughLinesP(
                (dark_mask.astype(np.uint8) * 255),
                rho=1, theta=np.pi / 180, threshold=10,
                minLineLength=8, maxLineGap=6
            )
            if lines_dark is not None and len(lines_dark) > 0:
                has_line = True
                line_score = max(line_score, min(0.85, 0.35 + len(lines_dark) * 0.05))

        instrument_detected = has_line or has_elongated_contour
        instrument_conf = max(line_score, contour_score)
        if not instrument_detected:
            instrument_conf = 0.35

        return instrument_detected, instrument_conf, (rx1, ry1, rx2, ry2)

    def _analyze_writing_motion(self, landmarks, w: int, h: int) -> Tuple[bool, float]:
        """Analyze temporal movement of index tip for writing or pointing gestures."""
        if landmarks is None or len(landmarks) < 9:
            self._history_xy.clear()
            return False, 0.0

        lm = landmarks[8]
        tx = lm.x * w if hasattr(lm, "x") else lm[0] * w
        ty = lm.y * h if hasattr(lm, "y") else lm[1] * h

        self._history_xy.append((tx, ty))
        if len(self._history_xy) > self._max_history:
            self._history_xy.pop(0)

        if len(self._history_xy) < 3:
            return True, 0.50

        xs = [pt[0] for pt in self._history_xy]
        ys = [pt[1] for pt in self._history_xy]

        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        total_std = math.hypot(std_x, std_y)

        dx = np.diff(xs)
        dy = np.diff(ys)

        sign_changes_x = np.count_nonzero(np.diff(np.sign(dx))) if len(dx) > 1 else 0
        sign_changes_y = np.count_nonzero(np.diff(np.sign(dy))) if len(dy) > 1 else 0
        total_oscillations = sign_changes_x + sign_changes_y

        is_oscillating = total_oscillations >= 1
        within_bounds = 0.5 <= total_std <= 55.0

        motion_detected = is_oscillating or within_bounds

        if within_bounds:
            std_factor = max(0.3, 1.0 - abs(total_std - 12.0) / 40.0)
            motion_conf = std_factor * (min(5.0, total_oscillations + 2) / 5.0)
        else:
            motion_conf = 0.25

        return motion_detected, float(motion_conf)

    def detect(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(detail="disabled")

        h, w = frame.shape[:2]
        context = context or {}
        landmarks = context.get("landmarks")

        if landmarks is None:
            self._history_xy.clear()
            self._smooth_score = 0.7 * self._smooth_score
            return DetectorResult(detected=False, confidence=0.0, detail="no_writing")

        # 1. Verify writing / pointing grip pose
        grip_ok, grip_conf = self._verify_writing_grip(landmarks, w, h)

        # 2. Detect pen/pencil/note instrument in hand ROI
        instrument_ok, instrument_conf, bbox = self._detect_writing_instrument(frame, landmarks, w, h)

        # 3. Verify writing / pointing motion
        motion_ok, motion_conf = self._analyze_writing_motion(landmarks, w, h)

        # Composite detection logic: grip is required, instrument or motion confirms
        is_writing = grip_ok and (instrument_ok or motion_ok)
        if is_writing:
            raw_score = (grip_conf * 0.40) + (instrument_conf * 0.35) + (motion_conf * 0.25)
        else:
            raw_score = 0.0

        # Temporal smoothing
        self._smooth_score = 0.50 * self._smooth_score + 0.50 * raw_score

        # Detection threshold
        effective_thr = max(0.28, float(self.confidence_threshold) - 0.12)
        detected = is_writing and (self._smooth_score >= effective_thr)
        detail = "Writing In Progress" if detected else "no_writing"

        return DetectorResult(
            detected=detected,
            confidence=float(self._smooth_score),
            detail=detail,
            bbox=bbox if detected else None,
        )
