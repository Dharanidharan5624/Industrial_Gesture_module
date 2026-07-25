"""Worker thread that runs the camera + monitoring pipeline off the UI thread.

Emits annotated frames and monitor state snapshots via Qt signals so the UI
stays responsive while inference runs in the background.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from typing import Optional

# Silence C++ level warnings from OpenCV, MediaPipe, and TensorFlow
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["FLAGS_stderrthreshold"] = "3"

import cv2
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

import numpy as np

from qt_compat import QtCore, Signal

from demo import GestureDetector
from monitoring_system import ScrewMonitor
from compliance.monitor import ComplianceMonitor
from compliance.logger import ComplianceLogger


class _suppress_stderr:
    """Context manager to suppress low-level C stderr outputs during camera probing."""
    def __enter__(self):
        try:
            self.err_fd = sys.stderr.fileno()
            self.saved_fd = os.dup(self.err_fd)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, self.err_fd)
        except Exception:
            self.saved_fd = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "saved_fd") and self.saved_fd is not None:
            try:
                os.dup2(self.saved_fd, self.err_fd)
                os.close(self.saved_fd)
                os.close(self.devnull)
            except Exception:
                pass

# Consecutive failed frame reads tolerated before treating the camera as
# actually disconnected. This absorbs the brief read hiccups that happen
# when a USB/webcam is unplugged and replugged, or when a cheap USB hub
# momentarily drops the device, so a hot-plug blip doesn't kill the session.
_MAX_CONSECUTIVE_READ_FAILURES = 20
_READ_RETRY_DELAY_SEC = 0.05

# A camera that just opened often isn't ready to deliver a real frame
# instantly (auto-exposure/white-balance settling, driver warm-up). Retry
# a few times with a short delay before concluding the device is actually
# unavailable, so scanning/opening doesn't false-negative a working camera.
_WARMUP_READ_ATTEMPTS = 5
_WARMUP_READ_DELAY_SEC = 0.15


def list_available_cameras(max_index: int = 8, skip_index: Optional[int] = None) -> list[int]:
    """Probe camera indices ``0..max_index`` and return the ones that open
    and deliver at least one frame.
    """
    available: list[int] = []
    with _suppress_stderr():
        for idx in range(max_index + 1):
            if idx == skip_index:
                continue
            cap = _open_capture(idx)
            if cap is not None:
                ok, frame = cap.read()
                if ok and frame is not None:
                    available.append(idx)
                cap.release()
    return available


def _open_capture(source: int) -> Optional[cv2.VideoCapture]:
    """Open ``source`` trying the backend most likely to respect the exact
    index requested, falling back to cv2.CAP_ANY. Returns an opened,
    frame-verified VideoCapture, or None if nothing worked.
    """
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    with _suppress_stderr():
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                for attempt in range(_WARMUP_READ_ATTEMPTS):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        return cap
                    if attempt < _WARMUP_READ_ATTEMPTS - 1:
                        time.sleep(_WARMUP_READ_DELAY_SEC)
            cap.release()
    return None


def grab_test_frame(source: int):
    """Open ``source``, grab exactly one frame, and release immediately.

    Used for a lightweight preview so the user can visually confirm which
    physical camera a given index actually maps to, without starting the
    full detection pipeline (and without fighting a running worker for the
    device — call this only when nothing else has the camera open).
    Returns the BGR frame (np.ndarray) or None if it couldn't be grabbed.
    """
    cap = _open_capture(source)
    if cap is None:
        return None
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    return frame


class CameraWorker(QtCore.QThread):
    """Background thread: read frames, run gesture + monitor, emit results."""

    frame_ready = Signal(object)        # annotated BGR frame (np.ndarray)
    state_ready = Signal(dict)          # monitor state snapshot
    gesture_ready = Signal(str, float, str)  # gesture, confidence, handedness
    compliance_ready = Signal(dict)     # compliance state snapshot
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
        compliance_settings: Optional[dict] = None,
        compliance_logger: Optional[ComplianceLogger] = None,
        operator_name: str = "",
        compliance_mock: bool = False,
    ):
        super().__init__(parent)
        self.config_path = config_path
        self.source = source
        self.worker_id = worker_id
        self.operator_name = operator_name
        self._running = False
        self._monitor: Optional[ScrewMonitor] = None
        self._compliance: Optional[ComplianceMonitor] = None
        self._compliance_settings = compliance_settings
        self._compliance_logger = compliance_logger
        self._compliance_mock = compliance_mock
        self._detector: Optional[GestureDetector] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.show_landmarks = True
        self.gesture_enabled = True
        self.monitor_enabled = True
        self.compliance_enabled = True
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
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Zero video queue latency
        # Push confidence threshold into config before creating detector
        if self.confidence_threshold:
            config.setdefault("detection", {})["confidence_threshold"] = self.confidence_threshold
            config.setdefault("mediapipe", {})["min_detection_confidence"] = self.confidence_threshold
            config.setdefault("mediapipe", {})["min_tracking_confidence"] = self.confidence_threshold

        self._detector = GestureDetector(config)
        self._monitor = ScrewMonitor(worker_id=self.worker_id)
        self._compliance = ComplianceMonitor(
            settings=self._compliance_settings,
            logger=self._compliance_logger,
            operator_id=self.worker_id,
            operator_name=self.operator_name,
            camera_id=str(self.source),
            mock=self._compliance_mock,
        )
        self._compliance.enabled = self.compliance_enabled
        self._running = True

        prev = time.time()
        consecutive_failures = 0
        while self._running:
            # Flush any stale queued frames in buffer so live feed has zero latency
            for _ in range(2):
                if not self._cap.grab():
                    break
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                ok, frame = self._cap.read()
            if not ok or frame is None:
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

            # Gesture detection (adds overlay on the same frame). Runs
            # whenever either "Show Landmarks Overlay" or "Gesture
            # Detection" is checked — both need the same underlying hand
            # detection, so the overlay checkbox alone is enough to see the
            # 21 landmark points + the recognized sign name.
            gesture, conf, handed = "no_gesture", 0.0, ""
            results = []
            # Also run hand detect when compliance is on — earbuds-in-palm needs landmarks.
            if self.gesture_enabled or self.show_landmarks or self.compliance_enabled:
                results = self._detector.detect(frame)
                if results:
                    top = results[0]
                    gesture, conf, handed = top.gesture, top.confidence, top.handedness
                    from demo import draw_results

                    # Only draw purple landmarks when the overlay checkbox is on.
                    if self.show_landmarks or self.gesture_enabled:
                        frame = draw_results(frame, results, show_landmarks=self.show_landmarks)
                    self.gesture_ready.emit(gesture, conf, handed)
                else:
                    self.gesture_ready.emit("no_gesture", 0.0, "")

            # Industrial monitoring.
            landmarks = results[0].landmarks if results else None
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

            # Operator compliance / safety monitoring (parallel to screw SOP).
            if self._compliance is not None and self.compliance_enabled:
                all_hand_lms = [res.landmarks for res in results] if results else []
                ctx = {
                    "landmarks": landmarks,
                    "hand_landmarks": all_hand_lms,
                    "hand_detected": bool(results),
                    "gesture": gesture,
                }
                cstate = self._compliance.process(frame, ctx)
                self.compliance_ready.emit(cstate.to_dict())

            fps = 1.0 / max(time.time() - prev, 1e-6)
            prev = time.time()
            # FPS first (top-right); compliance alerts are drawn under this row.
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

    def apply_compliance_settings(self, settings: dict) -> None:
        self._compliance_settings = settings
        self.compliance_enabled = bool(settings.get("enabled", True))
        if self._compliance is not None:
            self._compliance.apply_settings(settings)

    def set_operator_info(self, operator_id: str, operator_name: str = "") -> None:
        self.worker_id = operator_id
        self.operator_name = operator_name
        if self._compliance is not None:
            self._compliance.set_operator(operator_id, operator_name)
        if self._monitor is not None:
            self._monitor.worker_id = operator_id

    def _cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None