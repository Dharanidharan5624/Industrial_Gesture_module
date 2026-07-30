"""Spectacles detector using MediaPipe Face Mesh eye-region heuristics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from compliance.detectors.base import BaseDetector
from compliance.models import DetectorResult
from constants import COMPLIANCE_EVENT_SPECTACLES

try:
    import mediapipe as mp  # type: ignore

    _MP_OK = hasattr(mp, "solutions")
except Exception:
    mp = None
    _MP_OK = False


class SpectaclesDetector(BaseDetector):
    name = "spectacles"
    event_type = COMPLIANCE_EVENT_SPECTACLES

    # Face-mesh landmark indices around left / right eyes + nose bridge
    _LEFT_EYE = (33, 133, 159, 145)
    _RIGHT_EYE = (362, 263, 386, 374)
    _BRIDGE = (6, 168, 197)

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.45):
        super().__init__(enabled=enabled, confidence_threshold=confidence_threshold)
        self._face_mesh = None
        self._ema = 0.0
        if _MP_OK:
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as exc:
                print(f"[compliance] SpectaclesDetector FaceMesh init failed: {exc}")
                self.model_unavailable_logged = True

    def detect(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(detail="disabled")
        if self._face_mesh is None:
            if not self.model_unavailable_logged:
                print("[compliance] SpectaclesDetector model_unavailable")
                self.model_unavailable_logged = True
            return self.unavailable_result()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self._face_mesh.process(rgb)
        if not out.multi_face_landmarks:
            self._ema = 0.0
            return DetectorResult(detected=False, confidence=0.0, detail="no_face")

        lm = out.multi_face_landmarks[0].landmark
        score = self._glasses_score(lm, frame, w, h)
        self._ema = 0.45 * self._ema + 0.55 * score
        # Slightly stricter than default — eyebrows/lashes alone should not count.
        thr = max(0.52, float(self.confidence_threshold) + 0.08)
        detected = self._ema >= thr
        return DetectorResult(
            detected=detected,
            confidence=float(self._ema),
            detail="Spectacles Present" if detected else "Spectacles Not Present",
        )

    def _glasses_score(self, lm, frame: np.ndarray, w: int, h: int) -> float:
        """Score frames via eye-band edges + horizontal rim lines + bridge cue."""
        left = self._eye_band_score(lm, frame, w, h, self._LEFT_EYE)
        right = self._eye_band_score(lm, frame, w, h, self._RIGHT_EYE)
        bridge = self._bridge_score(lm, frame, w, h)

        # Need evidence on both eyes OR one strong eye + bridge (profile).
        if left < 0.25 and right < 0.25:
            return 0.0
        if max(left, right) < 0.35 and bridge < 0.20:
            return 0.0

        # Average the two eyes; boost when bridge shows a frame bar.
        eye = 0.5 * (left + right) if (left > 0.15 and right > 0.15) else max(left, right)
        score = 0.72 * eye + 0.28 * bridge
        return float(min(1.0, max(0.0, score)))

    def _eye_band_score(self, lm, frame, w, h, indices) -> float:
        xs = [int(lm[i].x * w) for i in indices]
        ys = [int(lm[i].y * h) for i in indices]
        if not xs:
            return 0.0
        pad_x, pad_y = 10, 14
        x1, x2 = max(0, min(xs) - pad_x), min(w, max(xs) + pad_x)
        y1, y2 = max(0, min(ys) - pad_y), min(h, max(ys) + pad_y)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Suppress soft skin texture; keep thin dark frame edges
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 70, 160)
        density = float(np.count_nonzero(edges)) / float(edges.size)

        # Horizontal rim evidence (top / bottom of lens)
        rh, rw = gray.shape
        top = edges[: max(2, rh // 3)]
        bot = edges[min(rh - 1, 2 * rh // 3) :]
        top_h = self._horizontal_line_strength(top)
        bot_h = self._horizontal_line_strength(bot)
        rim = 0.5 * (top_h + bot_h)

        # Dark thin strokes vs skin (frame material)
        dark = (gray < 90).astype(np.uint8)
        dark_ratio = float(np.count_nonzero(dark)) / float(dark.size)
        # Eyebrows alone: dark mass high, rims weak → damp
        if dark_ratio > 0.22 and rim < 0.15:
            density *= 0.45

        score = 0.55 * min(1.0, max(0.0, (density - 0.025) / 0.09)) + 0.45 * rim
        return float(min(1.0, max(0.0, score)))

    def _bridge_score(self, lm, frame, w, h) -> float:
        xs = [int(lm[i].x * w) for i in self._BRIDGE]
        ys = [int(lm[i].y * h) for i in self._BRIDGE]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        bw = max(16, int(0.045 * w))
        bh = max(10, int(0.030 * h))
        x1, x2 = max(0, cx - bw), min(w, cx + bw)
        y1, y2 = max(0, cy - bh), min(h, cy + bh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 70, 160)
        # Glasses bridge = short dark horizontal / diagonal bar
        return self._horizontal_line_strength(edges)

    @staticmethod
    def _horizontal_line_strength(edge_roi: np.ndarray) -> float:
        if edge_roi is None or edge_roi.size == 0:
            return 0.0
        # Count rows that look like thin horizontal strokes
        row = np.mean(edge_roi > 0, axis=1)
        if row.size == 0:
            return 0.0
        strong = float(np.count_nonzero(row > 0.12)) / float(row.size)
        # Prefer a few strong rows, not salt-and-pepper
        peak = float(np.max(row))
        return float(min(1.0, 0.6 * strong / 0.25 + 0.4 * peak / 0.35))
