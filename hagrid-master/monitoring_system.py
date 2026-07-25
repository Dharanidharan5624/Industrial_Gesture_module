"""Industrial screw-tightening workspace monitor.

Implements the real-time workflow checks described in the project spec:
safety glove detection, screwdriver alignment, rotation tracking and an
auto-logging system that captures screenshots and appends CSV rows (with
optional Supabase cloud sync).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import constants
from constants import (
    ALIGN_THRESHOLD_PX,
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_GLOVE,
    COLOR_NORMAL,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    CSV_COLUMNS,
    DEFAULT_WORKER_ID,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GLOVE_SKIN_RATIO_THRESHOLD,
    LOG_CSV_PATH,
    LOG_DIR,
    ROTATION_TURN_TARGET,
    SCREW_CENTER,
    SCREENSHOT_PREFIX,
)
from tool_recognizer import ToolRecognizer

# Optional MediaPipe import - the monitor works with landmarks when present
# but degrades gracefully (object-detection only) when it is not.
try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:  # pragma: no cover - sandbox without mediapipe
    mp = None
    _MP_OK = False


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _screenshot_name() -> str:
    return f"{SCREENSHOT_PREFIX}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


_SENTINEL_HAND = "--"   # sentinel value used when no hand is present

@dataclass
class HandInfo:
    handedness: str = "Right"
    glove: str = "Normal"          # "Glove" or "Normal"
    landmarks: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    is_detected: bool = False       # True only when a real hand is in frame


@dataclass
class MonitorState:
    aligned: bool = False
    direction: str = "Idle"        # "Clockwise" | "Anti-Clockwise" | "Idle"
    cumulative_angle: float = 0.0
    rotation_count: float = 0.0
    completed: bool = False
    screenshot_path: str = ""
    last_angle: Optional[float] = None
    card_visible: bool = False
    hand: HandInfo = field(default_factory=HandInfo)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Supabase sync (optional, best-effort)
# ---------------------------------------------------------------------------


def _sync_to_supabase(row: dict, table: str) -> None:
    try:
        from supabase_client import get_supabase  # local helper
    except Exception:
        return
    try:
        client = get_supabase()
        if client is not None:
            client.table(table).insert(row).execute()
    except Exception as exc:  # pragma: no cover - network/path dependent
        print(f"[Supabase] sync failed: {exc}")


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------


class ScrewMonitor:
    """Per-frame industrial monitoring engine."""

    def __init__(
        self,
        worker_id: str = DEFAULT_WORKER_ID,
        screw_center: Tuple[int, int] = SCREW_CENTER,
        align_threshold: int = ALIGN_THRESHOLD_PX,
        glove_ratio: float = GLOVE_SKIN_RATIO_THRESHOLD,
        turn_target: float = ROTATION_TURN_TARGET,
        log_csv: str = LOG_CSV_PATH,
        log_dir: str = LOG_DIR,
        reference_dir: str = "images",
        use_supabase: bool = True,
        supabase_table: str = "screw_monitoring_log",
    ):
        self.worker_id = worker_id
        self.screw_center = screw_center
        self.align_threshold = align_threshold
        self.glove_ratio = glove_ratio
        self.turn_target = turn_target
        self.log_csv = log_csv
        self.log_dir = log_dir
        self.use_supabase = use_supabase
        self.supabase_table = supabase_table
        self.state = MonitorState()
        self.show_screw_target: bool = False
        self.tool_recognizer = ToolRecognizer(reference_dir=reference_dir)
        # Bottom Y of the on-frame status panel (Aligned/Direction/…).
        # Compliance overlays stack below this so they don't cover Aligned.
        self._overlay_bottom_y = 50
        self._ensure_csv_header()
        self._hands = None
        if _MP_OK:
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )

    # -- setup -------------------------------------------------------------
    def _ensure_csv_header(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(CSV_COLUMNS)

    # -- per-frame pipeline ------------------------------------------------
    def process(
        self,
        frame: np.ndarray,
        ext_landmarks: np.ndarray = None,
        ext_handedness: str = None,
        detect_hand: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]
        if detect_hand or ext_landmarks is not None:
            self._update_hand(frame, ext_landmarks, ext_handedness)
        else:
            self.clear_hand()

        if self.state.hand.landmarks is not None:
            self._check_alignment(self.state.hand.landmarks, w, h)
            self._track_rotation(self.state.hand.landmarks)
        else:
            self.state.aligned = False
            self.state.direction = "Idle"

        self._draw_overlay(annotated, w, h)

        if self.state.rotation_count >= self.turn_target and not self.state.completed:
            self._finalize_operation(annotated)

        return annotated

    # -- hand update -------------------------------------------------------
    def clear_hand(self) -> None:
        # Use sentinel so UI knows no hand is present and shows "--"
        self.state.hand = HandInfo(
            handedness=_SENTINEL_HAND,
            glove=_SENTINEL_HAND,
            is_detected=False,
        )
        self.state.aligned = False
        self.state.direction = "Idle"

    def _flip_handedness(self, mp_label: str) -> str:
        """MediaPipe reports handedness mirrored (from subject's PoV).
        Flip it so the label matches what the *camera* sees (real-world L/R)."""
        if mp_label == "Left":
            return "Right"
        if mp_label == "Right":
            return "Left"
        return mp_label

    def _update_hand(self, frame: np.ndarray, ext_landmarks: np.ndarray = None, ext_handedness: str = None) -> None:
        if ext_landmarks is not None:
            lm = ext_landmarks
            # ext_handedness from demo.py is already flipped to real-world L/R; use as-is.
            handedness = ext_handedness or "Right"
            bbox = self._landmark_bbox(lm, frame.shape[1], frame.shape[0])
            glove = self._glove_check(frame, bbox)
            self.state.hand = HandInfo(handedness=handedness, glove=glove, landmarks=lm, bbox=bbox, is_detected=True)
            return

        if not _MP_OK or self._hands is None:
            self.clear_hand()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            self.clear_hand()
            return
        chosen = results.multi_hand_landmarks[0]
        raw_handed = "Right"
        if results.multi_handedness:
            raw_handed = results.multi_handedness[0].classification[0].label
        handedness = self._flip_handedness(raw_handed)
        lm = np.array([(p.x * frame.shape[1], p.y * frame.shape[0], p.z) for p in chosen.landmark], dtype=np.float32)
        bbox = self._landmark_bbox(lm, frame.shape[1], frame.shape[0])
        glove = self._glove_check(frame, bbox)
        self.state.hand = HandInfo(handedness=handedness, glove=glove, landmarks=lm, bbox=bbox, is_detected=True)

    @staticmethod
    def _landmark_bbox(lm: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        xs = np.clip(lm[:, 0], 0, w - 1).astype(int)
        ys = np.clip(lm[:, 1], 0, h - 1).astype(int)
        pad = 20
        x1, y1 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
        x2, y2 = min(int(xs.max()) + pad, w), min(int(ys.max()) + pad, h)
        return x1, y1, x2, y2

    # -- 5.2.1 glove check -------------------------------------------------
    def _glove_check(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> str:
        """Return 'Normal' (bare skin) or 'Glove' (glove detected).

        Uses a widened HSV skin-tone mask covering light to dark melanin.
        If the skin-pixel ratio is above the threshold the hand is bare (Normal).
        If skin pixels are rare the hand is covered by a glove.
        """
        if bbox is None:
            return "Normal"
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 5 or y2 - y1 < 5:
            return "Normal"
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Widened skin-tone HSV ranges to handle varied lighting & skin tones.
        # Low-hue (reddish-brown) band for most skin tones.
        lower1 = np.array([0,  20,  50], dtype=np.uint8)
        upper1 = np.array([25, 200, 255], dtype=np.uint8)
        # High-hue (wrap-around) band for reddish extremes.
        lower2 = np.array([160, 20,  50], dtype=np.uint8)
        upper2 = np.array([180, 200, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        skin_ratio = float(np.count_nonzero(mask)) / float(mask.size + 1e-6)
        self.state.confidence = max(self.state.confidence, (1.0 - skin_ratio) * 100.0)
        # skin_ratio HIGH  -> lots of skin pixels -> bare hand -> "Normal"
        # skin_ratio LOW   -> few  skin pixels   -> glove on  -> "Glove"
        return "Normal" if skin_ratio >= self.glove_ratio else "Glove"

    # -- 5.2.2 alignment ---------------------------------------------------
    def _check_alignment(self, lm: np.ndarray, w: int, h: int) -> None:
        index_mcp = lm[5][:2] * np.array([w, h])
        index_tip = lm[8][:2] * np.array([w, h])
        tip_point = index_tip.astype(int)
        dist = math.hypot(tip_point[0] - self.screw_center[0], tip_point[1] - self.screw_center[1])
        self.state.aligned = dist < self.align_threshold

    # -- 5.2.3 rotation tracking ------------------------------------------
    def _track_rotation(self, lm: np.ndarray) -> None:
        wrist = lm[0][:2]
        middle_mcp = lm[9][:2]
        angle = math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])
        if self.state.last_angle is not None:
            delta = angle - self.state.last_angle
            # Wrap to [-pi, pi].
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) < 1.0:  # ignore jitter spikes
                self.state.cumulative_angle += delta
                self.state.rotation_count = abs(self.state.cumulative_angle) / (2 * math.pi)
                if self.state.cumulative_angle > 0.05:
                    self.state.direction = "Clockwise"
                elif self.state.cumulative_angle < -0.05:
                    self.state.direction = "Anti-Clockwise"
        self.state.last_angle = angle

    # -- 5.2.4 auto logging ------------------------------------------------
    def _finalize_operation(self, frame: np.ndarray) -> None:
        self.state.completed = True
        self.state.card_visible = True
        path = os.path.join(self.log_dir, _screenshot_name())
        cv2.imwrite(path, frame)
        self.state.screenshot_path = path
        action = "Screw Tight" if self.state.direction == "Clockwise" else "Screw Loose"
        hand_label = f"{self.state.hand.handedness} Hand ({self.state.hand.glove})"
        row = {
            "Timestamp": _now(),
            "Worker_ID": self.worker_id,
            "Action": action,
            "Object": "Screw",
            "Tool": "Screwdriver",
            "Hand": hand_label,
            "Direction": self.state.direction,
            "Rotation_Count": round(self.state.rotation_count, 2),
            "Confidence": f"{self.state.confidence:.1f}%",
            "Status": "Completed",
            "Screenshot_Path": path,
        }
        with open(self.log_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writerow(row)
        if self.use_supabase:
            _sync_to_supabase(row, self.supabase_table)

    def reset(self) -> None:
        self.state = MonitorState()

    # -- drawing -----------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, w: int, h: int) -> None:
        # Screw target - only drawn when active SOP step is alignment / rotation
        if getattr(self, "show_screw_target", False):
            cv2.circle(frame, self.screw_center, 18, COLOR_BLUE, 2)
            cv2.putText(frame, "Screw", (self.screw_center[0] - 25, self.screw_center[1] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 1)

        # Hand bbox + handedness + glove
        hand = self.state.hand
        if hand.bbox is not None:
            x1, y1, x2, y2 = hand.bbox
            color = COLOR_GLOVE if hand.glove == "Glove" else COLOR_NORMAL
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{hand.handedness} Hand ({hand.glove})"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Alignment line
        if hand.landmarks is not None:
            index_mcp = hand.landmarks[5][:2] * np.array([w, h])
            index_tip = hand.landmarks[8][:2] * np.array([w, h])
            cv2.line(frame, index_mcp.astype(int), index_tip.astype(int),
                     COLOR_GREEN if self.state.aligned else COLOR_RED, 3)

        # Status panel — start below FPS overlay (drawn at y≈30) so lines never collide.
        panel = [
            f"Aligned: {'YES' if self.state.aligned else 'NO'}",
            f"Direction: {self.state.direction}",
            f"Turns: {self.state.rotation_count:.2f} / {self.turn_target}",
            f"Glove: {hand.glove}",
        ]
        y = 50
        for line in panel:
            cv2.rectangle(frame, (8, y - 4), (8 + 230, y + 18), COLOR_BLACK, -1)
            cv2.putText(frame, line, (12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
            y += 26
        # Expose end Y so compliance alerts can stack underneath without overlap.
        self._overlay_bottom_y = y

        # Completion card
        if self.state.card_visible:
            self._draw_completion_card(frame)

    def _draw_completion_card(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x1, y1 = w // 2 - 200, h // 2 - 90
        x2, y2 = w // 2 + 200, h // 2 + 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GREEN, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
        lines = [
            "OPERATION COMPLETED",
            f"Turns: {self.state.rotation_count:.2f}",
            f"Direction: {self.state.direction}",
            f"Hand: {self.state.hand.handedness} ({self.state.hand.glove})",
            "Press ENTER to reset",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x1 + 20, y1 + 35 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def run(source: int = 0, worker_id: str = DEFAULT_WORKER_ID) -> None:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    monitor = ScrewMonitor(worker_id=worker_id)
    print("[Monitor] press 'q' to quit, ENTER to reset after a completion card.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = monitor.process(frame)
        cv2.imshow("Industrial SOP Monitor", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 13 and monitor.state.card_visible:  # ENTER
            monitor.reset()
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial screw-tightening monitor")
    parser.add_argument("--source", type=int, default=constants.DEFAULT_SOURCE, help="Camera index or video path")
    parser.add_argument("--worker-id", type=str, default=DEFAULT_WORKER_ID)
    parser.add_argument("--no-supabase", action="store_true", help="Disable cloud log sync")
    args = parser.parse_args()
    run(source=args.source, worker_id=args.worker_id)


if __name__ == "__main__":
    main()
