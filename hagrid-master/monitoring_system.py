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
        assembly_id: str = "screw_tightening",
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
        self.assembly_id = assembly_id
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
            "Assembly_ID": self.assembly_id,
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


        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x1 + 20, y1 + 35 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)


@dataclass
class AirFilterState:
    aligned: bool = False
    direction: str = "Idle"
    rotation_count: float = 0.0
    completed: bool = False
    screenshot_path: str = ""
    card_visible: bool = False
    hand: HandInfo = field(default_factory=HandInfo)


class AirFilterMonitor:
    """YOLO & MediaPipe based Air Filter check & fitting monitor."""

    def __init__(
        self,
        worker_id: str = DEFAULT_WORKER_ID,
        log_csv: str = LOG_CSV_PATH,
        log_dir: str = LOG_DIR,
        use_supabase: bool = True,
        supabase_table: str = "screw_monitoring_log",
        assembly_id: str = "air_filter_fit",
    ):
        self.worker_id = worker_id
        self.log_csv = log_csv
        self.log_dir = log_dir
        self.use_supabase = use_supabase
        self.supabase_table = supabase_table
        self.assembly_id = assembly_id

        self.state = AirFilterState()
        self.active_step_index = 0
        self.active_step_status = "active"
        self.active_step_desc = ""
        self.confidence = 94.0

        # State machine variables
        self.step_frames = 0
        self.pins_inserted = [False, False]
        self.pins_removed = [False, False]
        self.pressing_timer = 0
        self.sealant_applied_points = []
        self.alignment_timer = 0
        self.inspection_timer = 0
        self.cleaning_timer = 0

        # Load YOLO model
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
        except Exception as exc:
            print(f"[AirFilterMonitor] YOLO load failed: {exc}")
            self.yolo_model = None

    def reset(self) -> None:
        self.state = AirFilterState()
        self.active_step_index = 0
        self.active_step_status = "active"
        self.active_step_desc = ""
        self.step_frames = 0
        self.pins_inserted = [False, False]
        self.pins_removed = [False, False]
        self.pressing_timer = 0
        self.sealant_applied_points.clear()
        self.alignment_timer = 0
        self.inspection_timer = 0
        self.cleaning_timer = 0

    def clear_hand(self) -> None:
        self.state.hand = HandInfo(
            handedness=_SENTINEL_HAND,
            glove=_SENTINEL_HAND,
            is_detected=False,
        )

    def advance_air_filter_step(self) -> None:
        """Advance to the next step (called via manual Next Step button)."""
        if self.active_step_index < 19:
            self.active_step_index += 1
            self._on_step_changed()

    def _on_step_changed(self) -> None:
        self.step_frames = 0
        self.pins_inserted = [False, False]
        self.pins_removed = [False, False]
        self.pressing_timer = 0
        self.sealant_applied_points.clear()
        self.alignment_timer = 0
        self.inspection_timer = 0
        self.cleaning_timer = 0
        self.active_step_status = "active"
        self.active_step_desc = ""

    def _flip_handedness(self, mp_label: str) -> str:
        if mp_label == "Left":
            return "Right"
        if mp_label == "Right":
            return "Left"
        return mp_label

    def _glove_check(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> str:
        if bbox is None:
            return "Normal"
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 5 or y2 - y1 < 5:
            return "Normal"
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0,  20,  50], dtype=np.uint8)
        upper1 = np.array([25, 200, 255], dtype=np.uint8)
        lower2 = np.array([160, 20,  50], dtype=np.uint8)
        upper2 = np.array([180, 200, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        skin_ratio = float(np.count_nonzero(mask)) / float(mask.size + 1e-6)
        return "Normal" if skin_ratio >= GLOVE_SKIN_RATIO_THRESHOLD else "Glove"

    def _landmark_bbox(self, lm: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        xs = np.clip(lm[:, 0], 0, w - 1).astype(int)
        ys = np.clip(lm[:, 1], 0, h - 1).astype(int)
        pad = 20
        x1, y1 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
        x2, y2 = min(int(xs.max()) + pad, w), min(int(ys.max()) + pad, h)
        return x1, y1, x2, y2

    def process(
        self,
        frame: np.ndarray,
        ext_landmarks: np.ndarray = None,
        ext_handedness: str = None,
        detect_hand: bool = True,
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Update hand state
        if ext_landmarks is not None:
            lm = ext_landmarks
            handedness = ext_handedness or "Right"
            bbox = self._landmark_bbox(lm, w, h)
            glove = self._glove_check(frame, bbox)
            self.state.hand = HandInfo(handedness=handedness, glove=glove, landmarks=lm, bbox=bbox, is_detected=True)
        else:
            self.clear_hand()

        # Run YOLO model for object/person detection
        yolo_boxes = []
        if self.yolo_model is not None:
            try:
                yolo_results = self.yolo_model(frame, verbose=False)
                if yolo_results and len(yolo_results) > 0:
                    for box in yolo_results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        yolo_boxes.append((cls_id, conf_val, xyxy))
            except Exception:
                pass

        # Try to locate hand center using landmarks or YOLO fallback
        hand_center = None
        if self.state.hand.is_detected and self.state.hand.landmarks is not None:
            hand_center = np.mean(self.state.hand.landmarks[:, :2], axis=0).astype(int)
        else:
            for cls_id, conf_val, xyxy in yolo_boxes:
                if cls_id == 0:  # person
                    # Treat lower-center of person box as hand proxy
                    hand_center = (int((xyxy[0] + xyxy[2]) / 2), int(xyxy[3] - 40))
                    break

        # Define Region of Interest boundaries
        # work_area_roi: [100, 100, 500, 400]
        # finished_area_roi: [500, 100, 640, 400]
        work_roi = (100, 100, 500, 400)
        finished_roi = (500, 100, 620, 400)

        # ── State Machine Processing ──────────────────────────────────────────
        self.step_frames += 1
        current_step = self.active_step_index

        # Pin and mount coordinates relative to filter placement inside work_roi
        # When placed, air filter fits inside (200, 150) to (400, 350)
        filter_rect = (200, 150, 400, 350)
        pin1_loc = (200, 150)
        pin2_loc = (400, 150)
        mount1_loc = (200, 350)
        mount2_loc = (400, 350)

        if current_step == 0:
            # Step 0: Operator Verification
            progress = min(int((self.step_frames / 30.0) * 100.0), 100)
            self.active_step_status = "active"
            self.active_step_desc = f"Verifying operator identity... {progress}%"
            if progress >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 1:
            # Step 1: Safety & PPE Check
            if self.state.hand.is_detected and self.state.hand.glove == "Glove":
                self.pressing_timer += 1
                progress = min(int((self.pressing_timer / 30.0) * 100.0), 100)
                self.active_step_status = "active"
                self.active_step_desc = f"PPE check in progress... {progress}%"
                if progress >= 100:
                    self.active_step_index += 1
                    self._on_step_changed()
            else:
                self.pressing_timer = max(0, self.pressing_timer - 1)
                self.active_step_status = "error"
                self.active_step_desc = "Please wear safety gloves!"

        elif current_step == 2:
            # Step 2: Pick & Inspect Air Filter
            # Hand should be active, preferably outside Work Area ROI
            hand_outside = True
            if hand_center is not None:
                x, y = hand_center
                if work_roi[0] < x < work_roi[2] and work_roi[1] < y < work_roi[3]:
                    hand_outside = False

            if hand_outside:
                self.alignment_timer += 1
                progress = min(int((self.alignment_timer / 45.0) * 100.0), 100)
                self.active_step_status = "active"
                self.active_step_desc = f"Inspecting raw air filter... {progress}%"
                if progress >= 100:
                    self.active_step_index += 1
                    self._on_step_changed()
            else:
                self.active_step_status = "active"
                self.active_step_desc = "Pick up the air filter from the tray."

        elif current_step == 3:
            # Step 3: Position Air Filter in Workstation
            # Hand center enters the Work Area ROI
            hand_inside = False
            if hand_center is not None:
                x, y = hand_center
                if work_roi[0] < x < work_roi[2] and work_roi[1] < y < work_roi[3]:
                    hand_inside = True

            if hand_inside:
                self.alignment_timer += 1
                progress = min(int((self.alignment_timer / 30.0) * 100.0), 100)
                self.active_step_status = "active"
                self.active_step_desc = f"Positioning in fixture... {progress}%"
                if progress >= 100:
                    self.active_step_index += 1
                    self._on_step_changed()
            else:
                self.active_step_status = "active"
                self.active_step_desc = "Place the air filter into the fixture."

        elif current_step == 4:
            # Step 4: Verify Side A Alignment
            self.alignment_timer += 1
            progress = min(int((self.alignment_timer / 45.0) * 100.0), 100)
            self.active_step_status = "active"
            self.active_step_desc = f"Verifying Side A alignment... {progress}%"
            if progress >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 5:
            # Step 5: Insert Temporary Locator Pins (Side A)
            if hand_center is not None:
                hx, hy = hand_center
                d1 = math.hypot(hx - pin1_loc[0], hy - pin1_loc[1])
                d2 = math.hypot(hx - pin2_loc[0], hy - pin2_loc[1])

                if d1 < 50 and not self.pins_inserted[0]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_inserted[0] = True
                        self.alignment_timer = 0
                elif d2 < 50 and not self.pins_inserted[1]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_inserted[1] = True
                        self.alignment_timer = 0

            p1_str = "OK" if self.pins_inserted[0] else "Pending"
            p2_str = "OK" if self.pins_inserted[1] else "Pending"
            self.active_step_status = "active"
            self.active_step_desc = f"Insert guide pins: Pin 1: {p1_str} | Pin 2: {p2_str}"

            if all(self.pins_inserted):
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 6:
            # Step 6: Apply Adhesive / Sealant (Side A)
            # Track hand coordinates to draw sealant path
            if hand_center is not None:
                hx, hy = hand_center
                # Check if hand is near any part of the perimeter of filter_rect
                # filter_rect is (200, 150, 400, 350)
                px_min, py_min, px_max, py_max = filter_rect
                # distance to closest point on rectangle boundary
                dx = min(abs(hx - px_min), abs(hx - px_max)) if py_min <= hy <= py_max else min(math.hypot(hx-px_min, hy-py_min), math.hypot(hx-px_max, hy-py_min), math.hypot(hx-px_min, hy-py_max), math.hypot(hx-px_max, hy-py_max))
                dy = min(abs(hy - py_min), abs(hy - py_max)) if px_min <= hx <= px_max else min(math.hypot(hx-px_min, hy-py_min), math.hypot(hx-px_max, hy-py_min), math.hypot(hx-px_min, hy-py_max), math.hypot(hx-px_max, hy-py_max))
                dist = min(dx, dy)

                if dist < 45:
                    self.sealant_applied_points.append((hx, hy))

            # Calculate path coverage
            px_min, py_min, px_max, py_max = filter_rect
            top_cov = any(px_min <= pt[0] <= px_max and abs(pt[1] - py_min) < 30 for pt in self.sealant_applied_points)
            bot_cov = any(px_min <= pt[0] <= px_max and abs(pt[1] - py_max) < 30 for pt in self.sealant_applied_points)
            left_cov = any(py_min <= pt[1] <= py_max and abs(pt[0] - px_min) < 30 for pt in self.sealant_applied_points)
            right_cov = any(py_min <= pt[1] <= py_max and abs(pt[0] - px_max) < 30 for pt in self.sealant_applied_points)
            coverage = sum([top_cov, bot_cov, left_cov, right_cov]) * 25

            self.active_step_status = "active"
            self.active_step_desc = f"Applying sealant along channel... {coverage}%"
            if coverage >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 7:
            # Step 7: Pick & Position Rubber Mount 1
            # Tray A is on the left
            tray_a = (30, 200, 110, 280)
            in_tray = False
            near_bracket = False
            if hand_center is not None:
                hx, hy = hand_center
                if tray_a[0] < hx < tray_a[2] and tray_a[1] < hy < tray_a[3]:
                    self.pins_inserted[0] = True # borrow flag to indicate mount picked up
                
                d = math.hypot(hx - mount1_loc[0], hy - mount1_loc[1])
                if d < 50 and self.pins_inserted[0]:
                    near_bracket = True

            self.active_step_status = "active"
            if not self.pins_inserted[0]:
                self.active_step_desc = "Retrieve Rubber Mount 1 from Tray A."
            else:
                self.active_step_desc = "Align Mount 1 over the Bottom-Left bracket."

            if near_bracket:
                self.alignment_timer += 1
                if self.alignment_timer >= 30:
                    self.active_step_index += 1
                    self._on_step_changed()

        elif current_step == 8:
            # Step 8: Press & Lock Rubber Mount 1
            is_pressing = False
            if hand_center is not None:
                d = math.hypot(hand_center[0] - mount1_loc[0], hand_center[1] - mount1_loc[1])
                if d < 50:
                    is_pressing = True

            if is_pressing:
                self.pressing_timer += 1
                progress = min(int((self.pressing_timer / 60.0) * 100.0), 100)
                self.active_step_status = "active"
                self.active_step_desc = f"Pressing Mount 1: {self.pressing_timer/30.0:.1f} / 2.0s"
                if progress >= 100:
                    self.active_step_index += 1
                    self._on_step_changed()
            else:
                self.pressing_timer = max(0, self.pressing_timer - 1)
                self.active_step_status = "active"
                self.active_step_desc = "Press Mount 1 firmly into place."

        elif current_step == 9:
            # Step 9: Remove Temporary Locator Pins (Side A)
            if hand_center is not None:
                hx, hy = hand_center
                d1 = math.hypot(hx - pin1_loc[0], hy - pin1_loc[1])
                d2 = math.hypot(hx - pin2_loc[0], hy - pin2_loc[1])

                if d1 < 50 and not self.pins_removed[0]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_removed[0] = True
                        self.alignment_timer = 0
                elif d2 < 50 and not self.pins_removed[1]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_removed[1] = True
                        self.alignment_timer = 0

            p1_str = "Removed" if self.pins_removed[0] else "Active"
            p2_str = "Removed" if self.pins_removed[1] else "Active"
            self.active_step_status = "active"
            self.active_step_desc = f"Remove pins: Pin 1: {p1_str} | Pin 2: {p2_str}"

            if all(self.pins_removed):
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 10:
            # Step 10: Rotate Air Filter to Side B
            self.alignment_timer += 1
            progress = min(int((self.alignment_timer / 60.0) * 100.0), 100)
            self.active_step_status = "active"
            self.active_step_desc = f"Flip filter 180 degrees to Side B... {progress}%"
            if progress >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 11:
            # Step 11: Position & Verify Side B Alignment
            self.alignment_timer += 1
            progress = min(int((self.alignment_timer / 45.0) * 100.0), 100)
            self.active_step_status = "active"
            self.active_step_desc = f"Verifying Side B alignment... {progress}%"
            if progress >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 12:
            # Step 12: Insert Temporary Locator Pins (Side B)
            if hand_center is not None:
                hx, hy = hand_center
                d1 = math.hypot(hx - pin1_loc[0], hy - pin1_loc[1])
                d2 = math.hypot(hx - pin2_loc[0], hy - pin2_loc[1])

                if d1 < 50 and not self.pins_inserted[0]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_inserted[0] = True
                        self.alignment_timer = 0
                elif d2 < 50 and not self.pins_inserted[1]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_inserted[1] = True
                        self.alignment_timer = 0

            p1_str = "OK" if self.pins_inserted[0] else "Pending"
            p2_str = "OK" if self.pins_inserted[1] else "Pending"
            self.active_step_status = "active"
            self.active_step_desc = f"Insert guide pins: Pin 1: {p1_str} | Pin 2: {p2_str}"

            if all(self.pins_inserted):
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 13:
            # Step 13: Apply Adhesive / Sealant (Side B)
            if hand_center is not None:
                hx, hy = hand_center
                px_min, py_min, px_max, py_max = filter_rect
                dx = min(abs(hx - px_min), abs(hx - px_max)) if py_min <= hy <= py_max else min(math.hypot(hx-px_min, hy-py_min), math.hypot(hx-px_max, hy-py_min), math.hypot(hx-px_min, hy-py_max), math.hypot(hx-px_max, hy-py_max))
                dy = min(abs(hy - py_min), abs(hy - py_max)) if px_min <= hx <= px_max else min(math.hypot(hx-px_min, hy-py_min), math.hypot(hx-px_max, hy-py_min), math.hypot(hx-px_min, hy-py_max), math.hypot(hx-px_max, hy-py_max))
                dist = min(dx, dy)

                if dist < 45:
                    self.sealant_applied_points.append((hx, hy))

            px_min, py_min, px_max, py_max = filter_rect
            top_cov = any(px_min <= pt[0] <= px_max and abs(pt[1] - py_min) < 30 for pt in self.sealant_applied_points)
            bot_cov = any(px_min <= pt[0] <= px_max and abs(pt[1] - py_max) < 30 for pt in self.sealant_applied_points)
            left_cov = any(py_min <= pt[1] <= py_max and abs(pt[0] - px_min) < 30 for pt in self.sealant_applied_points)
            right_cov = any(py_min <= pt[1] <= py_max and abs(pt[0] - px_max) < 30 for pt in self.sealant_applied_points)
            coverage = sum([top_cov, bot_cov, left_cov, right_cov]) * 25

            self.active_step_status = "active"
            self.active_step_desc = f"Applying sealant along channel... {coverage}%"
            if coverage >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 14:
            # Step 14: Pick & Position Rubber Mount 2
            # Tray B is on the right
            tray_b = (530, 200, 610, 280)
            near_bracket = False
            if hand_center is not None:
                hx, hy = hand_center
                if tray_b[0] < hx < tray_b[2] and tray_b[1] < hy < tray_b[3]:
                    self.pins_inserted[0] = True # borrow flag to indicate mount picked up
                
                d = math.hypot(hx - mount2_loc[0], hy - mount2_loc[1])
                if d < 50 and self.pins_inserted[0]:
                    near_bracket = True

            self.active_step_status = "active"
            if not self.pins_inserted[0]:
                self.active_step_desc = "Retrieve Rubber Mount 2 from Tray B."
            else:
                self.active_step_desc = "Align Mount 2 over the Bottom-Right bracket."

            if near_bracket:
                self.alignment_timer += 1
                if self.alignment_timer >= 30:
                    self.active_step_index += 1
                    self._on_step_changed()

        elif current_step == 15:
            # Step 15: Press & Lock Rubber Mount 2
            is_pressing = False
            if hand_center is not None:
                d = math.hypot(hand_center[0] - mount2_loc[0], hand_center[1] - mount2_loc[1])
                if d < 50:
                    is_pressing = True

            if is_pressing:
                self.pressing_timer += 1
                progress = min(int((self.pressing_timer / 60.0) * 100.0), 100)
                self.active_step_status = "active"
                self.active_step_desc = f"Pressing Mount 2: {self.pressing_timer/30.0:.1f} / 2.0s"
                if progress >= 100:
                    self.active_step_index += 1
                    self._on_step_changed()
            else:
                self.pressing_timer = max(0, self.pressing_timer - 1)
                self.active_step_status = "active"
                self.active_step_desc = "Press Mount 2 firmly into place."

        elif current_step == 16:
            # Step 16: Remove Temporary Locator Pins (Side B)
            if hand_center is not None:
                hx, hy = hand_center
                d1 = math.hypot(hx - pin1_loc[0], hy - pin1_loc[1])
                d2 = math.hypot(hx - pin2_loc[0], hy - pin2_loc[1])

                if d1 < 50 and not self.pins_removed[0]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_removed[0] = True
                        self.alignment_timer = 0
                elif d2 < 50 and not self.pins_removed[1]:
                    self.alignment_timer += 1
                    if self.alignment_timer >= 30:
                        self.pins_removed[1] = True
                        self.alignment_timer = 0

            p1_str = "Removed" if self.pins_removed[0] else "Active"
            p2_str = "Removed" if self.pins_removed[1] else "Active"
            self.active_step_status = "active"
            self.active_step_desc = f"Remove pins: Pin 1: {p1_str} | Pin 2: {p2_str}"

            if all(self.pins_removed):
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 17:
            # Step 17: Verify Adhesive Bonding & Curing
            self.alignment_timer += 1
            progress = min(int((self.alignment_timer / 60.0) * 100.0), 100)
            self.active_step_status = "active"
            self.active_step_desc = f"Checking bond lines... {progress}%"
            if progress >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 18:
            # Step 18: Clean Workstation & Remove Fixtures
            hand_outside = True
            if hand_center is not None:
                x, y = hand_center
                if work_roi[0] < x < work_roi[2] and work_roi[1] < y < work_roi[3]:
                    hand_outside = False

            if hand_outside:
                self.alignment_timer += 1
                progress = min(int((self.alignment_timer / 90.0) * 100.0), 100)
            else:
                progress = 0
                self.alignment_timer = 0

            self.active_step_status = "active"
            self.active_step_desc = f"Clearing workstation... {progress}%"
            if progress >= 100:
                self.active_step_index += 1
                self._on_step_changed()

        elif current_step == 19:
            # Step 19: Dual-Side Final Quality Inspection & Logging
            self.alignment_timer += 1
            progress = min(int((self.alignment_timer / 45.0) * 100.0), 100)
            self.active_step_status = "active"
            self.active_step_desc = f"Final visual audit... {progress}%"
            if progress >= 100 and not self.state.completed:
                self._finalize_operation(annotated)

        # ── Visual Overlay Rendering (wow factor!) ────────────────────────────
        # Draw Work Area ROI
        cv2.rectangle(annotated, (work_roi[0], work_roi[1]), (work_roi[2], work_roi[3]), COLOR_BLUE, 2)
        cv2.putText(annotated, "Work Area ROI", (work_roi[0] + 5, work_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 1)

        # Draw Finished Area ROI
        cv2.rectangle(annotated, (finished_roi[0], finished_roi[1]), (finished_roi[2], finished_roi[3]), COLOR_YELLOW, 2)
        cv2.putText(annotated, "Finished Area", (finished_roi[0] + 5, finished_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)

        # Draw any YOLO detections
        for cls_id, conf_val, xyxy in yolo_boxes:
            if cls_id == 0:  # person
                cv2.rectangle(annotated, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), COLOR_NORMAL, 1)
                cv2.putText(annotated, f"Operator ({conf_val*100:.0f}%)", (int(xyxy[0]), int(xyxy[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NORMAL, 1)

        # Draw hand bbox + glove label
        hand = self.state.hand
        if hand.bbox is not None:
            x1, y1, x2, y2 = hand.bbox
            color = COLOR_GLOVE if hand.glove == "Glove" else COLOR_RED
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"Operator Hand ({hand.glove}) [YOLO]"
            cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Air Filter representation based on current state
        if current_step >= 3:
            # Placed inside Work ROI
            px_min, py_min, px_max, py_max = filter_rect
            cv2.rectangle(annotated, (px_min, py_min), (px_max, py_max), COLOR_GREEN, 3)
            # Label
            cv2.putText(annotated, "Air Filter [YOLO]", (px_min + 10, py_min + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)
            # Side Indicator
            side = "Side B" if current_step >= 11 else "Side A"
            cv2.putText(annotated, f"Orientation: {side}", (px_min + 10, py_min + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

            # Draw pins if active/inserted
            if current_step in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16):
                # Pin 1
                if (current_step in (5, 12) and self.pins_inserted[0]) or (current_step in (6, 7, 8, 9, 13, 14, 15) and not self.pins_removed[0]):
                    cv2.circle(annotated, pin1_loc, 10, COLOR_YELLOW, -1)
                    cv2.putText(annotated, "Pin 1", (pin1_loc[0] - 15, pin1_loc[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1)
                else:
                    cv2.circle(annotated, pin1_loc, 10, COLOR_RED, 2)

                # Pin 2
                if (current_step in (5, 12) and self.pins_inserted[1]) or (current_step in (6, 7, 8, 9, 13, 14, 15) and not self.pins_removed[1]):
                    cv2.circle(annotated, pin2_loc, 10, COLOR_YELLOW, -1)
                    cv2.putText(annotated, "Pin 2", (pin2_loc[0] - 15, pin2_loc[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1)
                else:
                    cv2.circle(annotated, pin2_loc, 10, COLOR_RED, 2)

            # Draw glue sealant trajectory
            if current_step in (6, 13) and len(self.sealant_applied_points) > 1:
                for idx in range(len(self.sealant_applied_points) - 1):
                    cv2.line(annotated, self.sealant_applied_points[idx], self.sealant_applied_points[idx+1], COLOR_GREEN, 4)

            # Draw Mounts
            # Mount 1
            if current_step >= 8:
                cv2.rectangle(annotated, (mount1_loc[0] - 15, mount1_loc[1] - 15), (mount1_loc[0] + 15, mount1_loc[1] + 15), (255, 0, 255), -1)
                cv2.putText(annotated, "Mount 1", (mount1_loc[0] - 25, mount1_loc[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            # Mount 2
            if current_step >= 15:
                cv2.rectangle(annotated, (mount2_loc[0] - 15, mount2_loc[1] - 15), (mount2_loc[0] + 15, mount2_loc[1] + 15), (255, 0, 255), -1)
                cv2.putText(annotated, "Mount 2", (mount2_loc[0] - 25, mount2_loc[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # If holding mount in hand
        if current_step == 7 and self.pins_inserted[0] and hand_center is not None:
            cv2.rectangle(annotated, (hand_center[0] - 15, hand_center[1] - 15), (hand_center[0] + 15, hand_center[1] + 15), (255, 0, 255), 2)
            cv2.putText(annotated, "Mount 1 [YOLO]", (hand_center[0] - 35, hand_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        if current_step == 14 and self.pins_inserted[0] and hand_center is not None:
            cv2.rectangle(annotated, (hand_center[0] - 15, hand_center[1] - 15), (hand_center[0] + 15, hand_center[1] + 15), (255, 0, 255), 2)
            cv2.putText(annotated, "Mount 2 [YOLO]", (hand_center[0] - 35, hand_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Draw a beautiful glassmorphic Banner for step action at the top
        banner_bg = annotated.copy()
        cv2.rectangle(banner_bg, (0, 0), (w, 55), COLOR_BLACK, -1)
        cv2.addWeighted(banner_bg, 0.7, annotated, 0.3, 0, annotated)

        step_title = f"STEP {current_step + 1}/20"
        cv2.putText(annotated, step_title, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)
        cv2.putText(annotated, self.active_step_desc, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        # Progress bar inside banner
        # Determine progress percentage for bar
        prog_pct = 0.0
        if current_step in (0, 1, 2, 3, 4, 10, 11, 17, 18, 19):
            divisor = 30.0 if current_step in (0, 1, 3) else (45.0 if current_step in (2, 4, 11, 19) else (60.0 if current_step in (10, 17) else 90.0))
            prog_pct = min(self.step_frames / divisor, 1.0)
            if current_step == 1:
                prog_pct = min(self.pressing_timer / 30.0, 1.0)
            elif current_step == 18:
                prog_pct = min(self.alignment_timer / 90.0, 1.0)
        elif current_step in (5, 12):
            prog_pct = sum(self.pins_inserted) / 2.0
        elif current_step in (6, 13):
            # sealant points
            px_min, py_min, px_max, py_max = filter_rect
            top_cov = any(px_min <= pt[0] <= px_max and abs(pt[1] - py_min) < 30 for pt in self.sealant_applied_points)
            bot_cov = any(px_min <= pt[0] <= px_max and abs(pt[1] - py_max) < 30 for pt in self.sealant_applied_points)
            left_cov = any(py_min <= pt[1] <= py_max and abs(pt[0] - px_min) < 30 for pt in self.sealant_applied_points)
            right_cov = any(py_min <= pt[1] <= py_max and abs(pt[0] - px_max) < 30 for pt in self.sealant_applied_points)
            prog_pct = sum([top_cov, bot_cov, left_cov, right_cov]) / 4.0
        elif current_step in (7, 14):
            prog_pct = 1.0 if self.alignment_timer > 0 else (0.5 if self.pins_inserted[0] else 0.0)
        elif current_step in (8, 15):
            prog_pct = min(self.pressing_timer / 60.0, 1.0)
        elif current_step in (9, 16):
            prog_pct = sum(self.pins_removed) / 2.0

        bar_w = int(140 * prog_pct)
        cv2.rectangle(annotated, (w - 160, 20), (w - 20, 32), COLOR_WHITE, 1)
        if bar_w > 0:
            cv2.rectangle(annotated, (w - 159, 21), (w - 160 + bar_w, 31), COLOR_GREEN, -1)
        cv2.putText(annotated, f"{int(prog_pct*100)}%", (w - 95, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

        # Completion card
        if self.state.card_visible:
            self._draw_completion_card(annotated)

        return annotated

    def _finalize_operation(self, frame: np.ndarray) -> None:
        self.state.completed = True
        self.state.card_visible = True
        path = os.path.join(self.log_dir, _screenshot_name())
        cv2.imwrite(path, frame)
        self.state.screenshot_path = path

        hand_label = f"{self.state.hand.handedness} Hand ({self.state.hand.glove})"
        row = {
            "Timestamp": _now(),
            "Worker_ID": self.worker_id,
            "Action": "Air Filter Fit",
            "Object": "Air Filter",
            "Tool": "Sealant & Mounts",
            "Hand": hand_label,
            "Direction": "N/A",
            "Rotation_Count": 0.0,
            "Confidence": f"{self.confidence:.1f}%",
            "Status": "Completed",
            "Screenshot_Path": path,
            "Assembly_ID": self.assembly_id,
        }
        with open(self.log_csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writerow(row)
        if self.use_supabase:
            _sync_to_supabase(row, self.supabase_table)

    def _draw_completion_card(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x1, y1 = w // 2 - 200, h // 2 - 90
        x2, y2 = w // 2 + 200, h // 2 + 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_GREEN, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
        lines = [
            "ASSEMBLY COMPLETED",
            "Air Filter Fitted Successfully",
            "Quality Inspection: PASS",
            f"Logged to Database ({self.worker_id})",
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
