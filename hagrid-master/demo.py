"""Real-time gesture detection via MediaPipe + geometric rule-based classifier.

Provides ``GestureDetector`` and ``draw_results`` consumed by camera_worker.py.
Degrades gracefully (no hand detection) when MediaPipe is unavailable.
"""

from __future__ import annotations

# Monkey patch protobuf MessageFactory to prevent MediaPipe import error in newer protobuf versions.
try:
    from google.protobuf import message_factory
    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        def GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)
        message_factory.MessageFactory.GetPrototype = GetPrototype
except Exception as exc:
    print(f"Failed to apply protobuf patch: {exc}")

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

import constants
from custom_utils.gesture_classifier import (
    classify_gesture,
    handedness_label,
)

# ---------------------------------------------------------------------------
# Optional MediaPipe import
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp          # type: ignore
    _MP_HANDS = mp.solutions.hands
    _MP_DRAWING = mp.solutions.drawing_utils
    _MP_OK = True
except Exception:                   # pragma: no cover
    mp = None
    _MP_HANDS = None
    _MP_DRAWING = None
    _MP_OK = False

# Sentinel used by ScrewMonitor / status panel when no hand is present.
_SENTINEL = "--"

# How many consecutive "no hand" frames before we emit no_gesture.
_GRACE_FRAMES = 3


# ---------------------------------------------------------------------------
# Data class emitted by GestureDetector per hand
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    gesture: str
    confidence: float
    handedness: str          # "Left" or "Right" (camera-corrected)
    landmarks: np.ndarray    # shape (21, 3) in pixel coords
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------

class GestureDetector:
    """Wraps MediaPipe Hands + geometric classifier.

    Falls back to returning an empty list (no detection) when MediaPipe is
    not installed or initialisation fails, so the rest of the pipeline can
    keep running without crashing.
    """

    def __init__(self, config: dict):
        mp_cfg = config.get("mediapipe", {})
        self._min_det  = float(mp_cfg.get("min_detection_confidence", 0.50))
        self._min_trk  = float(mp_cfg.get("min_tracking_confidence",  0.50))
        self._max_hands = int(mp_cfg.get("max_num_hands", 2))
        self._mirror    = bool(mp_cfg.get("mirror", True))

        self._hands = None
        if _MP_OK:
            try:
                self._hands = _MP_HANDS.Hands(
                    model_complexity=0,
                    static_image_mode=False,
                    max_num_hands=self._max_hands,
                    min_detection_confidence=self._min_det,
                    min_tracking_confidence=self._min_trk,
                )
            except Exception as exc:
                print(f"[GestureDetector] MediaPipe init failed: {exc}")
                self._hands = None

        self._no_result_count = 0

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run detection on a BGR frame.  Returns a (possibly empty) list."""
        if self._hands is None or frame is None:
            return []

        h, w = frame.shape[:2]
        # MediaPipe expects RGB; optionally flip for front-facing camera
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._mirror:
            rgb = cv2.flip(rgb, 1)

        try:
            mp_results = self._hands.process(rgb)
        except Exception:
            return []

        if not mp_results.multi_hand_landmarks:
            return []

        results: List[DetectionResult] = []
        multi_lm = mp_results.multi_hand_landmarks
        multi_hand = mp_results.multi_handedness or []

        for i, hand_lm in enumerate(multi_lm):
            # ── landmark array (21, 3) in pixel coordinates ──────────────
            lm_arr = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lm.landmark],
                dtype=np.float32,
            )
            # Undo the flip on x so coordinates are in the *original* frame's
            # pixel space, not the mirrored inference space.
            if self._mirror:
                lm_arr[:, 0] = w - lm_arr[:, 0]

            # ── bounding box ─────────────────────────────────────────────
            pad = 12
            x1 = max(0,     int(lm_arr[:, 0].min()) - pad)
            y1 = max(0,     int(lm_arr[:, 1].min()) - pad)
            x2 = min(w - 1, int(lm_arr[:, 0].max()) + pad)
            y2 = min(h - 1, int(lm_arr[:, 1].max()) + pad)

            # ── handedness (camera-corrected) ─────────────────────────────
            if i < len(multi_hand):
                raw_side = multi_hand[i].classification[0].label
                # MediaPipe labels are from the *selfie* perspective.
                # If we mirrored the frame before inference, the labels are
                # already correct; if not, we need to invert them.
                side = raw_side if self._mirror else (
                    "Right" if raw_side == "Left" else "Left"
                )
            else:
                side = "Right"

            # ── gesture classification ────────────────────────────────────
            gesture = classify_gesture(lm_arr)
            conf = 0.85   # geometric classifier doesn't produce a probability

            results.append(DetectionResult(
                gesture=gesture,
                confidence=conf,
                handedness=side,
                landmarks=lm_arr,
                bbox=(x1, y1, x2, y2),
            ))

        return results

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

# Landmark connection topology (mirrors MediaPipe's HAND_CONNECTIONS)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

_COLOR_LANDMARK  = (148, 0, 211)   # purple dots
_COLOR_CONNECTION = (100, 0, 180)  # darker purple lines
_COLOR_BBOX       = (0, 200, 0)    # green bounding box
_COLOR_WHITE      = (255, 255, 255)


def draw_results(
    frame: np.ndarray,
    results: List[DetectionResult],
    show_landmarks: bool = True,
) -> np.ndarray:
    """Draw bounding boxes, gesture labels, and (optionally) landmarks."""
    out = frame.copy()
    h, w = out.shape[:2]

    for res in results:
        x1, y1, x2, y2 = res.bbox

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_BBOX, 2)

        # Label badge
        label = f"{res.handedness} | {res.gesture} ({res.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (30, 30, 30), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_WHITE, 1, cv2.LINE_AA)

        if show_landmarks and res.landmarks is not None:
            lm = res.landmarks
            # Connections
            for a, b in _HAND_CONNECTIONS:
                pt_a = (int(np.clip(lm[a, 0], 0, w - 1)),
                        int(np.clip(lm[a, 1], 0, h - 1)))
                pt_b = (int(np.clip(lm[b, 0], 0, w - 1)),
                        int(np.clip(lm[b, 1], 0, h - 1)))
                cv2.line(out, pt_a, pt_b, _COLOR_CONNECTION, 1, cv2.LINE_AA)
            # Dots
            for idx in range(21):
                px = int(np.clip(lm[idx, 0], 0, w - 1))
                py = int(np.clip(lm[idx, 1], 0, h - 1))
                cv2.circle(out, (px, py), 3, _COLOR_LANDMARK, -1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Standalone demo runner (optional, for testing without the GUI)
# ---------------------------------------------------------------------------

def _run_demo(config_path: str, source: int) -> None:
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    detector = GestureDetector(config)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[demo] Could not open camera {source}")
        return

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        results = detector.detect(frame)
        if results:
            frame = draw_results(frame, results, show_landmarks=True)
        cv2.imshow("Gesture Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture detection demo")
    parser.add_argument("-p", "--config", default="configs/gesture.yaml")
    parser.add_argument("--source", type=int, default=0)
    args = parser.parse_args()
    _run_demo(args.config, args.source)
