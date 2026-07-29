"""Worker thread that runs the camera + monitoring pipeline off the UI thread.

Emits annotated frames and monitor state snapshots via Qt signals so the UI
stays responsive while inference runs in the background.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from qt_compat import QtCore, Signal

from demo import GestureDetector
from monitoring_system import ScrewMonitor

# Consecutive failed frame reads tolerated before treating the camera as
# actually disconnected. This absorbs the brief read hiccups that happen
# when a USB/webcam is unplugged and replugged, or when a cheap USB hub
# momentarily drops the device, so a hot-plug blip doesn't kill the session.
_MAX_CONSECUTIVE_READ_FAILURES = 20
_READ_RETRY_DELAY_SEC = 0.05


def list_available_cameras(max_index: int = 8) -> list[int]:
    """Probe camera indices ``0..max_index`` and return the ones that open
    and deliver at least one frame.

    Index 0 is conventionally the system's default/built-in camera; any
    index >= 1 is whatever external USB/webcam the OS has enumerated after
    it. This probe doesn't care which is which — it just reports what's
    actually usable right now, which is what the UI needs to populate a
    camera picker or to validate a selection before starting the worker.
    """
    available: list[int] = []
    for idx in range(max_index + 1):
        cap = _open_capture(idx)
        if cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                available.append(idx)
            cap.release()
    return available


def _open_capture(source: int) -> Optional[cv2.VideoCapture]:
    """Open ``source`` trying the backends most likely to work well for a
    given OS, falling back to cv2.CAP_ANY. Returns an opened VideoCapture,
    or None if nothing worked.

    Using an OS-appropriate backend (DirectShow/MSMF on Windows, V4L2 on
    Linux) rather than always taking whatever cv2.CAP_ANY happens to pick
    is what makes switching between the built-in camera and an external
    USB camera reliable across platforms, and gives more consistent
    behavior when a camera is hot-plugged.
    """
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


class CameraWorker(QtCore.QThread):
    """Background thread: read frames, run gesture + monitor, emit results."""

    frame_ready = Signal(object)        # annotated BGR frame (np.ndarray)
    state_ready = Signal(dict)          # monitor state snapshot
    gesture_ready = Signal(str, float, str)  # gesture, confidence, handedness
    error = Signal(str)
    # Emitted when the requested source can't be opened at all, along with
    # whatever camera indices ARE currently usable, so the UI can offer them.
    camera_unavailable = Signal(int, list)

    def __init__(
        self,
        config_path: str = "configs/gesture.yaml",
        source: int = 0,
        worker_id: str = "EMP001",
        parent=None,
    ):
        super().__init__(parent)
        self.config_path = config_path
        self.source = source
        self.worker_id = worker_id
        self._running = False
        self._monitor: Optional[ScrewMonitor] = None
        self._detector: Optional[GestureDetector] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.show_landmarks = True
        self.gesture_enabled = True
        self.monitor_enabled = True
        # Camera / AI settings pushed from Settings page
        self.cam_width: int = 640
        self.cam_height: int = 480
        self.rotation: int = 0          # 0 | 90 | 180 | 270
        self.confidence_threshold: float = 0.50
        self.rot_sensitivity: float = 5.0

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:
        import yaml

        try:
            with open(self.config_path) as fh:
                config = yaml.safe_load(fh)
        except Exception as exc:
            self.error.emit(f"Config load failed: {exc}")
            return

        self._cap = _open_capture(self.source)
        if self._cap is None or not self._cap.isOpened():
            available = list_available_cameras()
            if available:
                msg = (
                    f"Camera index {self.source} is unavailable (disconnected, "
                    f"in use by another app, or doesn't exist). "
                    f"Available cameras: {available}."
                )
            else:
                msg = (
                    f"Camera index {self.source} is unavailable, and no other "
                    f"cameras were detected. Check the connection and try again."
                )
            self.error.emit(msg)
            self.camera_unavailable.emit(self.source, available)
            return
        # Use worker-level overrides if set, else fall back to YAML
        cfg_cam = config.get("camera", {})
        w = self.cam_width or cfg_cam.get("width", 640)
        h = self.cam_height or cfg_cam.get("height", 480)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Push confidence threshold into config before creating detector
        if self.confidence_threshold:
            config.setdefault("detection", {})["confidence_threshold"] = self.confidence_threshold
            config.setdefault("mediapipe", {})["min_detection_confidence"] = self.confidence_threshold
            config.setdefault("mediapipe", {})["min_tracking_confidence"] = self.confidence_threshold

        self._detector = GestureDetector(config)
        self._monitor = ScrewMonitor(worker_id=self.worker_id)
        self._running = True

        prev = time.time()
        consecutive_failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                    self.error.emit(
                        f"Camera {self.source} stopped responding "
                        f"(likely unplugged or released by the OS)."
                    )
                    self.camera_unavailable.emit(self.source, list_available_cameras())
                    break
                time.sleep(_READ_RETRY_DELAY_SEC)
                continue
            consecutive_failures = 0

            # Gesture detection (adds overlay on the same frame).
            gesture, conf, handed = "no_gesture", 0.0, ""
            results = []
            if self.gesture_enabled:
                results = self._detector.detect(frame)
                if results:
                    top = results[0]
                    gesture, conf, handed = top.gesture, top.confidence, top.handedness
                    from demo import draw_results

                    frame = draw_results(frame, results, show_landmarks=self.show_landmarks)
                    self.gesture_ready.emit(gesture, conf, handed)
                else:
                    self.gesture_ready.emit("no_gesture", 0.0, "")

            # Industrial monitoring.
            if self.monitor_enabled:
                if results:
                    top = results[0]
                    frame = self._monitor.process(frame, top.landmarks, top.handedness)
                else:
                    self._monitor.clear_hand()
                    frame = self._monitor.process(frame, detect_hand=False)
                self.state_ready.emit(self._state_snapshot(gesture, conf, handed))
            else:
                self._monitor.clear_hand()
                self.state_ready.emit(self._state_snapshot("no_gesture", 0.0, ""))

            fps = 1.0 / max(time.time() - prev, 1e-6)
            prev = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            # Apply rotation if set
            rot = self.rotation
            if rot == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            self.frame_ready.emit(frame)

        self._cleanup()

    def _state_snapshot(self, gesture: str, conf: float, handed: str) -> dict:
        st = self._monitor.state
        hand = st.hand
        hand_detected = getattr(hand, "is_detected", False)
        return {
            "aligned": st.aligned,
            "direction": st.direction,
            "rotation_count": round(st.rotation_count, 2),
            "turn_target": self._monitor.turn_target,
            # Use "--" sentinel when no hand is in frame
            "glove": hand.glove if hand_detected else "--",
            "handedness": hand.handedness if hand_detected else "--",
            "hand_detected": hand_detected,
            "gesture": gesture,
            "gesture_confidence": conf,
            "completed": st.completed,
            "card_visible": st.card_visible,
            "screenshot_path": st.screenshot_path,
        }

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def reset_monitor(self) -> None:
        if self._monitor is not None:
            self._monitor.reset()

    def _cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None