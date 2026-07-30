"""Real-time gesture detection demo.

Uses MediaPipe Hands for landmark extraction and hand localisation, then
classifies the gesture with either a deep-learning backbone (when a
checkpoint is available) or the geometric rule-based fallback classifier.

Usage:
    python demo.py -p configs/gesture.yaml --landmarks
"""

from __future__ import annotations

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
    LandmarkFrame,
    classify_gesture,
    finger_states,
    handedness_label,
)
from custom_utils.utils import frame_to_tensor, load_checkpoint, topk_prediction
from models.gesture_net import GestureNet

try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:  # pragma: no cover
    mp = None
    _MP_OK = False

try:
    import torch
    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_OK = False


@dataclass
class GestureResult:
    gesture: str
    confidence: float
    handedness: str
    bbox: Optional[Tuple[int, int, int, int]]
    landmarks: Optional[np.ndarray]


class GestureDetector:
    """Orchestrates MediaPipe + DL/ geometric classification."""

    def __init__(self, config: dict):
        self.config = config
        self.stable_frames = config.get("detection", {}).get("stable_frames", 3)
        self.fallback_threshold = config.get("detection", {}).get("fallback_threshold", 0.4)
        self.conf_threshold = config.get("detection", {}).get("confidence_threshold", 0.6)
        self.show_landmarks = config.get("ui", {}).get("show_landmarks", True)

        # Deep learning backbone (optional).
        model_cfg = config.get("model", {})
        self.net: Optional[GestureNet] = None
        if _TORCH_OK and model_cfg.get("name"):
            self.net = GestureNet(
                model_name=model_cfg["name"],
                checkpoint=model_cfg.get("checkpoint") or None,
                device="cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu",
                pretrained=False,
            )

        # MediaPipe Hands.
        self.hands = None
        if _MP_OK:
            mp_cfg = config.get("mediapipe", {})
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=mp_cfg.get("static_image_mode", False),
                max_num_hands=mp_cfg.get("max_num_hands", 2),
                model_complexity=mp_cfg.get("model_complexity", 1),
                min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
                min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
            )
        self._stable: List[str] = []

    def detect(self, frame: np.ndarray) -> List[GestureResult]:
        results: List[GestureResult] = []
        if not _MP_OK or self.hands is None:
            return results
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.hands.process(rgb)
        if not mp_results.multi_hand_landmarks:
            return results
        h, w = frame.shape[:2]
        for idx, hand_lms in enumerate(mp_results.multi_hand_landmarks):
            lm = np.array([(p.x * w, p.y * h, p.z) for p in hand_lms.landmark], dtype=np.float32)
            handed = "Right"
            if mp_results.multi_handedness and idx < len(mp_results.multi_handedness):
                mp_label = handedness_label(mp_results.multi_handedness[idx].classification[0].label)
                # MediaPipe labels from subject's PoV (mirrored). Flip for real-world camera view.
                handed = "Left" if mp_label == "Right" else "Right"
            bbox = self._bbox(lm, w, h)
            gesture, conf = self._classify(frame, lm, bbox)
            results.append(GestureResult(gesture, conf, handed, bbox, lm))
        return results

    def _classify(self, frame: np.ndarray, lm: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[str, float]:
        # Try the DL backbone first when available.
        if self.net is not None and self.net.available and bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                res = self.net.predict(crop)
                if res is not None:
                    idx, prob, name = res
                    if prob / 100.0 >= self.fallback_threshold:
                        return name, prob
        # Geometric fallback.
        gesture = classify_gesture(lm)
        return gesture, 95.0

    @staticmethod
    def _bbox(lm: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        xs = np.clip(lm[:, 0], 0, w - 1).astype(int)
        ys = np.clip(lm[:, 1], 0, h - 1).astype(int)
        pad = 20
        return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
                min(int(xs.max()) + pad, w), min(int(ys.max()) + pad, h))

    # -- smoothing ---------------------------------------------------------
    def smooth(self, gesture: str) -> str:
        self._stable.append(gesture)
        if len(self._stable) > self.stable_frames:
            self._stable.pop(0)
        if not self._stable:
            return gesture
        return max(set(self._stable), key=self._stable.count)



# Friendly display names and sub-descriptions for each gesture.
GESTURE_DISPLAY: dict = {
    "palm":          ("Palm",          "Open"),
    "fist":          ("Fist",          "Closed"),
    "one":           ("One",           "Point Up"),
    "peace":         ("Peace",         "V Sign"),
    "three":         ("Three",         "3 Fingers"),
    "four":          ("Four",          "4 Fingers"),
    "ok":            ("OK",            "Circle"),
    "like":          ("Like",          "Thumbs Up"),
    "dislike":       ("Dislike",       "Thumbs Down"),
    "rock":          ("Rock",          "Horns"),
    "call":          ("Call",          "Phone Sign"),
    "middle_finger": ("Mid. Finger",   "Vertical"),
    "little_finger": ("Pinkie",        "Little Finger"),
    "thumb_index":   ("Gun",           "Thumb Index"),
    "no_gesture":    ("No Gesture",    ""),
}

# BGR colour per gesture for the banner
GESTURE_COLORS: dict = {
    "palm":          (180, 40, 130),
    "fist":          (50,  80, 200),
    "one":           (0,  160, 220),
    "peace":         (0,  180, 80),
    "three":         (0,  140, 200),
    "four":          (20, 120, 200),
    "ok":            (0,  200, 120),
    "like":          (30, 180, 30),
    "dislike":       (0,   60, 200),
    "rock":          (120, 20, 200),
    "call":          (200, 80, 0),
    "middle_finger": (0,   40, 200),
    "little_finger": (200, 100, 0),
    "thumb_index":   (180, 40, 0),
    "no_gesture":    (80,  80, 80),
}


def _draw_gesture_banner(
    frame: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    color_bgr: tuple,
) -> None:
    """Draw a pill-shaped semi-transparent banner centred at (cx, cy)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 14, 8
    x1 = cx - tw // 2 - pad_x
    y1 = cy - th - pad_y
    x2 = cx + tw // 2 + pad_x
    y2 = cy + pad_y

    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)

    # Semi-transparent filled rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # White text
    tx = cx - tw // 2
    ty = y2 - pad_y
    cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_results(frame: np.ndarray, results: List[GestureResult], show_landmarks: bool = True) -> np.ndarray:
    for r in results:
        gesture = r.gesture or "no_gesture"
        color = GESTURE_COLORS.get(gesture, (100, 100, 100))

        # --- Bounding box ---
        if r.bbox is not None:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # --- Pill banner label ---
            disp_name, description = GESTURE_DISPLAY.get(gesture, (gesture.replace("_", " ").title(), ""))
            hand_side = r.handedness  # "Left" or "Right"
            if description:
                label_text = f"{hand_side} Hand: {disp_name} ({description})"
            else:
                label_text = f"{hand_side} Hand: {disp_name}"

            banner_cx = (x1 + x2) // 2
            banner_cy = max(y1 - 16, 30)
            _draw_gesture_banner(frame, label_text, banner_cx, banner_cy, color)

        # --- Landmarks (dots on each joint) ---
        if show_landmarks and r.landmarks is not None:
            for pt in r.landmarks:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, color, -1)

    return frame




def run(config_path: str, source: int = 0, landmarks: bool = False) -> None:
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    detector = GestureDetector(config)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("camera", {}).get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("camera", {}).get("height", 480))
    print("[demo] press 'q' to quit.")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = detector.detect(frame)
        frame = draw_results(frame, results, show_landmarks=landmarks)
        fps = 1.0 / max(time.time() - prev, 1e-6)
        prev = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, constants.COLOR_GREEN, 2)
        cv2.imshow("HAGRID Gesture Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="HAGRID real-time gesture demo")
    parser.add_argument("-p", "--config", default="configs/gesture.yaml", help="Path to gesture YAML config")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--landmarks", action="store_true", help="Draw MediaPipe landmarks")
    args = parser.parse_args()
    run(args.config, source=args.source, landmarks=args.landmarks)


if __name__ == "__main__":
    main()
