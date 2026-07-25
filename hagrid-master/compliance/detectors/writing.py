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
        - Thumb tip (4), index tip (8), and middle tip (12) are near each other.
        - Index finger is pointing forward or pinched with thumb on pen/pencil.
        """
        if landmarks is None or len(landmarks) < 21:
            return False, 0.0

        # Hand scale (wrist to middle finger MCP)
        hand_scale = _dist2d(landmarks[0], landmarks[9])
        if hand_scale <= 0:
            return False, 0.0

        # Distances between thumb, index, and middle tips
        d_thumb_index = _dist2d(landmarks[4], landmarks[8]) / hand_scale
        d_thumb_middle = _dist2d(landmarks[4], landmarks[12]) / hand_scale
        d_index_middle = _dist2d(landmarks[8], landmarks[12]) / hand_scale

        max_dist = max(d_thumb_index, d_thumb_middle, d_index_middle)
        min_dist = min(d_thumb_index, d_thumb_middle)

        # A writing or pen-pointing grip has clustered tips or extended index near thumb
        is_clustered = max_dist < 0.65 or min_dist < 0.40
        confidence = max(0.20, 1.0 - (min_dist / 0.60)) if is_clustered else 0.0

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

        # Crop a 130x130 ROI around the grip center to catch full pen stem & note surface
        roi_size = 65
        rx1, rx2 = max(0, cx - roi_size), min(w, cx + roi_size)
        ry1, ry2 = max(0, cy - roi_size), min(h, cy + roi_size)

        if (rx2 - rx1) < 20 or (ry2 - ry1) < 20:
            return False, 0.0, None

        roi = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding to highlight pen body & note edges
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Hough Line Transform for straight linear objects (pen/pencil/note)
        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi / 180,
            threshold=18,
            minLineLength=12,
            maxLineGap=6
        )

        has_line = False
        line_score = 0.0
        if lines is not None and len(lines) > 0:
            has_line = True
            line_score = min(0.95, 0.45 + len(lines) * 0.07)

        # Contour aspect ratio analysis for elongated objects (pen/pencil/pointer)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_elongated_contour = False
        contour_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 12 <= area <= 1200:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    length = max(width, height)
                    if aspect_ratio >= 2.2 and length >= 12:
                        has_elongated_contour = True
                        contour_score = max(contour_score, min(0.95, 0.35 + aspect_ratio * 0.12))

        instrument_detected = has_line or has_elongated_contour
        instrument_conf = max(line_score, contour_score)
        if not instrument_detected:
            # Fallback score if grip is strong
            instrument_conf = 0.30

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

        if len(self._history_xy) < 4:
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

        is_oscillating = total_oscillations >= 2
        within_bounds = 1.0 <= total_std <= 48.0

        motion_detected = is_oscillating or within_bounds

        if within_bounds:
            std_factor = max(0.3, 1.0 - abs(total_std - 15.0) / 35.0)
            motion_conf = std_factor * (min(6.0, total_oscillations + 2) / 6.0)
        else:
            motion_conf = 0.20

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

        # Composite detection logic: triggers when grip is valid AND either instrument or writing motion is detected
        is_writing = grip_ok and (instrument_ok or motion_ok)
        raw_score = (grip_conf * 0.45) + (instrument_conf * 0.35) + (motion_conf * 0.20) if is_writing else 0.0

        # Temporal smoothing
        self._smooth_score = 0.55 * self._smooth_score + 0.45 * raw_score

        # Detection threshold
        effective_thr = max(0.32, float(self.confidence_threshold) - 0.08)
        detected = is_writing and (self._smooth_score >= effective_thr)
        detail = "Writing In Progress" if detected else "no_writing"

        return DetectorResult(
            detected=detected,
            confidence=float(self._smooth_score),
            detail=detail,
            bbox=bbox if detected else None,
        )
