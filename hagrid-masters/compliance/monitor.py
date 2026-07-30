"""ComplianceMonitor — orchestrates detectors, debounce, events, overlays."""

from __future__ import annotations

import copy
import datetime as _dt
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from compliance.detectors import (
    EarbudsDetector,
    PhoneDetector,
    ShirtButtonDetector,
    SpectaclesDetector,
    WritingDetector,
)
from compliance.logger import ComplianceLogger
from compliance.models import ComplianceEvent, ComplianceState, DetectorResult
from compliance.notifier import is_warning, message_for
from constants import (
    COLOR_RED,
    COLOR_YELLOW,
    COMPLIANCE_DEFAULT_SETTINGS,
    COMPLIANCE_EVENT_BLUETOOTH,
    COMPLIANCE_EVENT_EARBUDS,
    COMPLIANCE_EVENT_MOBILE_PHONE,
    COMPLIANCE_EVENT_PASSED,
    COMPLIANCE_EVENT_SHIRT_BUTTON,
    COMPLIANCE_EVENT_SPECTACLES,
    COMPLIANCE_EVENT_WRITING,
    COMPLIANCE_FRAME_SKIP,
)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ComplianceMonitor:
    """Parallel safety/compliance pipeline used by CameraWorker."""

    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        logger: Optional[ComplianceLogger] = None,
        operator_id: str = "EMP001",
        operator_name: str = "",
        camera_id: str = "0",
        mock: bool = False,
    ):
        cfg = copy.deepcopy(COMPLIANCE_DEFAULT_SETTINGS)
        if settings:
            cfg.update({k: v for k, v in settings.items() if k != "detectors"})
            if "detectors" in settings:
                cfg["detectors"] = {**cfg.get("detectors", {}), **settings["detectors"]}
        self.settings = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.45))
        self.cooldown_sec = float(cfg.get("warning_cooldown_sec", 8.0))
        self.save_evidence = bool(cfg.get("save_evidence_images", True))
        self.notify_on_pass = bool(cfg.get("notify_on_pass", False))
        self.frame_skip = COMPLIANCE_FRAME_SKIP
        self._frame_i = 0
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.camera_id = str(camera_id)
        self.logger = logger or ComplianceLogger()
        self.mock = mock

        dets = cfg.get("detectors", {})
        self.phone = PhoneDetector(enabled=dets.get("mobile_phone", True), confidence_threshold=self.confidence_threshold)
        self.shirt = ShirtButtonDetector(enabled=dets.get("shirt_button", True), confidence_threshold=self.confidence_threshold)
        self.earbuds = EarbudsDetector(enabled=dets.get("bluetooth_earbuds", True), confidence_threshold=self.confidence_threshold)
        self.spectacles = SpectaclesDetector(enabled=dets.get("spectacles", True), confidence_threshold=self.confidence_threshold)
        self.writing = WritingDetector(enabled=dets.get("writing", True), confidence_threshold=self.confidence_threshold)

        self._last_fire: Dict[str, float] = {}
        self._warning_count = 0
        self._last_state = ComplianceState()
        self._mock_tick = 0

    # -- runtime config -----------------------------------------------------
    def apply_settings(self, settings: Dict[str, Any]) -> None:
        if "enabled" in settings:
            self.enabled = bool(settings["enabled"])
        if "confidence_threshold" in settings:
            self.confidence_threshold = float(settings["confidence_threshold"])
            for d in (self.phone, self.shirt, self.earbuds, self.spectacles, self.writing):
                d.set_confidence_threshold(self.confidence_threshold)
        if "warning_cooldown_sec" in settings:
            self.cooldown_sec = float(settings["warning_cooldown_sec"])
        if "save_evidence_images" in settings:
            self.save_evidence = bool(settings["save_evidence_images"])
        if "notify_on_pass" in settings:
            self.notify_on_pass = bool(settings["notify_on_pass"])
        dets = settings.get("detectors") or {}
        if "mobile_phone" in dets:
            self.phone.set_enabled(bool(dets["mobile_phone"]))
        if "shirt_button" in dets:
            self.shirt.set_enabled(bool(dets["shirt_button"]))
        if "bluetooth_earbuds" in dets:
            self.earbuds.set_enabled(bool(dets["bluetooth_earbuds"]))
        if "spectacles" in dets:
            self.spectacles.set_enabled(bool(dets["spectacles"]))
        if "writing" in dets:
            self.writing.set_enabled(bool(dets["writing"]))

    def set_operator(self, operator_id: str, operator_name: str = "") -> None:
        self.operator_id = operator_id
        self.operator_name = operator_name

    def set_camera_id(self, camera_id: str) -> None:
        self.camera_id = str(camera_id)

    # -- main loop ----------------------------------------------------------
    def process(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> ComplianceState:
        if not self.enabled:
            st = ComplianceState(warning_count=self._warning_count)
            self._last_state = st
            return st

        self._frame_i += 1
        if self._frame_i % max(1, self.frame_skip) != 0:
            # Zero-lag optimization: reuse cached detector results on skipped frames
            cached = copy.deepcopy(self._last_state)
            cached.new_events = []
            if not self.mock:
                self._draw_overlays(frame, cached)
            return cached

        context = context or {}
        if self.mock:
            state = self._mock_process()
        else:
            state = self._real_process(frame, context)

        state.warning_count = self._warning_count
        state.active_alerts = self._alerts_from(state)
        self._last_state = state

        # Draw detection overlays on frame first so saved evidence screenshots include red boxes and labels
        self._draw_overlays(frame, state)

        events = self._emit_events(frame, state)
        state.new_events = events
        return state

    def _real_process(self, frame: np.ndarray, context: Dict[str, Any]) -> ComplianceState:
        phone = self.phone.detect(frame, context)
        shirt = self.shirt.detect(frame, context)
        buds = self.earbuds.detect(frame, context)
        specs = self.spectacles.detect(frame, context)
        writing = self.writing.detect(frame, context)

        bluetooth = DetectorResult()
        earbuds = DetectorResult()
        if buds.detected:
            if getattr(self.earbuds, "last_kind", "earbuds") == "bluetooth":
                bluetooth = buds
            else:
                earbuds = buds

        return ComplianceState(
            phone=phone,
            shirt_button_open=shirt if shirt.detected else DetectorResult(
                detected=False, confidence=shirt.confidence, detail=shirt.detail
            ),
            earbuds=earbuds,
            bluetooth=bluetooth,
            spectacles=specs,
            writing=writing,
        )

    def _mock_process(self) -> ComplianceState:
        """Synthetic detections for UI QA without models."""
        self._mock_tick += 1
        t = self._mock_tick
        phone = DetectorResult(detected=(t % 40 == 10), confidence=0.92, detail="Mobile phone detected")
        shirt = DetectorResult(detected=(t % 55 == 20), confidence=0.81, detail="Shirt button open")
        earbuds = DetectorResult(detected=(t % 70 == 30), confidence=0.77, detail="Earbuds Detected")
        bluetooth = DetectorResult(detected=False)
        spectacles = DetectorResult(
            detected=True, confidence=0.88, detail="Spectacles Present"
        ) if (t % 5 != 0) else DetectorResult(detected=False, confidence=0.2, detail="Spectacles Not Present")
        writing = DetectorResult(detected=(t % 90 == 45), confidence=0.7, detail="Writing In Progress")
        return ComplianceState(
            phone=phone,
            shirt_button_open=shirt,
            earbuds=earbuds,
            bluetooth=bluetooth,
            spectacles=spectacles,
            writing=writing,
        )

    def _alerts_from(self, state: ComplianceState) -> List[str]:
        alerts = []
        if state.phone.detected:
            alerts.append(message_for(COMPLIANCE_EVENT_MOBILE_PHONE, state.phone.detail))
        if state.shirt_button_open.detected:
            alerts.append(message_for(COMPLIANCE_EVENT_SHIRT_BUTTON, state.shirt_button_open.detail))
        if state.earbuds.detected:
            alerts.append(message_for(COMPLIANCE_EVENT_EARBUDS, state.earbuds.detail))
        if state.bluetooth.detected:
            alerts.append(message_for(COMPLIANCE_EVENT_BLUETOOTH, state.bluetooth.detail))
        if state.writing.detected:
            alerts.append(message_for(COMPLIANCE_EVENT_WRITING, state.writing.detail))
        if state.spectacles.detected:
            alerts.append(message_for(COMPLIANCE_EVENT_SPECTACLES, state.spectacles.detail))
        return alerts

    def _cooldown_ok(self, event_type: str) -> bool:
        import time

        now = time.time()
        last = self._last_fire.get(event_type, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self._last_fire[event_type] = now
        return True

    def _emit_events(self, frame: np.ndarray, state: ComplianceState) -> List[ComplianceEvent]:
        candidates = [
            (COMPLIANCE_EVENT_MOBILE_PHONE, state.phone, "Warning"),
            (COMPLIANCE_EVENT_SHIRT_BUTTON, state.shirt_button_open, "Warning"),
            (COMPLIANCE_EVENT_EARBUDS, state.earbuds, "Warning"),
            (COMPLIANCE_EVENT_BLUETOOTH, state.bluetooth, "Warning"),
            (COMPLIANCE_EVENT_WRITING, state.writing, "Info"),
            (COMPLIANCE_EVENT_SPECTACLES, state.spectacles, "Info"),
        ]
        events: List[ComplianceEvent] = []
        any_warning = False
        for event_type, result, status in candidates:
            if not result.detected:
                continue
            if not self._cooldown_ok(event_type):
                continue
            shot = ""
            if self.save_evidence and status == "Warning":
                shot = self.logger.save_evidence(frame, event_type)
            evt = ComplianceEvent(
                timestamp=_now(),
                operator_id=self.operator_id,
                operator_name=self.operator_name,
                event_type=event_type,
                detection_result=message_for(event_type, result.detail),
                camera_id=self.camera_id,
                confidence=float(result.confidence),
                screenshot_path=shot,
                status=status,
            )
            self.logger.log_event(evt)
            if status == "Warning":
                self._warning_count += 1
                any_warning = True
            events.append(evt)

        if (
            self.notify_on_pass
            and not any_warning
            and not state.active_alerts
            and self._cooldown_ok(COMPLIANCE_EVENT_PASSED)
        ):
            # Only when explicitly enabled — quiet by default
            if not any(
                r.detected
                for r in (
                    state.phone,
                    state.shirt_button_open,
                    state.earbuds,
                    state.bluetooth,
                )
            ):
                evt = ComplianceEvent(
                    timestamp=_now(),
                    operator_id=self.operator_id,
                    operator_name=self.operator_name,
                    event_type=COMPLIANCE_EVENT_PASSED,
                    detection_result=message_for(COMPLIANCE_EVENT_PASSED),
                    camera_id=self.camera_id,
                    confidence=1.0,
                    status="Passed",
                )
                self.logger.log_event(evt)
                events.append(evt)
        return events

    def _draw_overlays(self, frame: np.ndarray, state: ComplianceState) -> None:
        # Draw detection boxes (earbuds / bluetooth: only the ear(s) that actually have a device).
        ear_boxes = []
        buds_det = getattr(self, "earbuds", None)
        if buds_det is not None and (state.earbuds.detected or state.bluetooth.detected):
            side_hint = str(getattr(buds_det, "last_side", "") or "")
            kind_hint = str(getattr(buds_det, "last_kind", "earbuds") or "earbuds")
            if getattr(buds_det, "last_left_bbox", None):
                tag = "left" if side_hint in ("left", "both", "") else side_hint or "left"
                if side_hint == "both":
                    tag = "left"
                # Hand-held path stores box in last_left_bbox with empty side
                if side_hint == "" and not getattr(buds_det, "last_right_bbox", None):
                    tag = ""
                ear_boxes.append((tag, kind_hint, buds_det.last_left_bbox))
            if getattr(buds_det, "last_right_bbox", None):
                ear_boxes.append(("right", kind_hint, buds_det.last_right_bbox))
        for side, kind_hint, box in ear_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, COLOR_RED, -1)
            prefix = "Bluetooth" if kind_hint == "bluetooth" else "Earbuds"
            if side == "left":
                label = f"{prefix} (L)"
            elif side == "right":
                label = f"{prefix} (R)"
            else:
                label = prefix
            self._draw_box_label(frame, label, x1, y1)

        # Phone: tight YOLO-style box + center point + label (never full-frame).
        phone_box = None
        phone_det = getattr(self, "phone", None)
        if state.phone.detected:
            if phone_det is not None and getattr(phone_det, "last_bbox", None):
                phone_box = phone_det.last_bbox
            elif state.phone.bbox:
                phone_box = state.phone.bbox
        if phone_box is not None:
            x1, y1, x2, y2 = phone_box
            fh, fw = frame.shape[:2]
            # Guard: skip accidental full-screen boxes
            if (x2 - x1) * (y2 - y1) <= 0.30 * fw * fh:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, COLOR_RED, -1)
                self._draw_box_label(frame, "Phone", x1, y1)

        # Shirt button: YOLO-style box
        # RED + label  → shirt is OPEN  (false positive risk → only when detected=True)
        # Tiny GREEN indicator → hand is near placket, shirt is correctly buttoned
        # NOTHING     → no interaction / no detection (clean frame)
        shirt_det = getattr(self, "shirt", None)
        if state.shirt_button_open.detected:
            shirt_bbox = None
            if shirt_det is not None and getattr(shirt_det, "last_bbox", None):
                shirt_bbox = shirt_det.last_bbox
            elif state.shirt_button_open.bbox:
                shirt_bbox = state.shirt_button_open.bbox
            if shirt_bbox:
                x1, y1, x2, y2 = shirt_bbox
                fh, fw = frame.shape[:2]
                if (x2 - x1) * (y2 - y1) <= 0.55 * fw * fh:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
                    cx_b, cy_b = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx_b, cy_b), 4, COLOR_RED, -1)
                    shirt_label = getattr(shirt_det, "last_label", "Shirt Button Open") if shirt_det else "Shirt Button Open"
                    self._draw_box_label(frame, shirt_label, x1, y1)
        elif shirt_det is not None:
            # Show a small GREEN indicator pill only when hand is actively near placket
            hand_conf = getattr(shirt_det, "last_conf", 0.0)
            bbox = getattr(shirt_det, "last_bbox", None)
            if hand_conf > 0.20 and bbox is not None:
                x1, y1, x2, y2 = bbox
                COLOR_GREEN = (0, 200, 60)
                # Draw only the top edge to indicate "scanning" without alarming
                cv2.line(frame, (x1, y1), (x2, y1), COLOR_GREEN, 2)
                cv2.putText(frame, "Shirt Button OK", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_GREEN, 2)

        # Bluetooth / writing boxes
        for result in (state.bluetooth, state.writing):
            if result.detected and result.bbox:
                x1, y1, x2, y2 = result.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
                if state.bluetooth.detected and result is state.bluetooth:
                    self._draw_box_label(frame, "Bluetooth", x1, y1)

        # Fallback single bbox for earbuds only if per-ear boxes missing
        if state.earbuds.detected and state.earbuds.bbox and not ear_boxes:
            x1, y1, x2, y2 = state.earbuds.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RED, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, COLOR_RED, -1)
            self._draw_box_label(frame, "Earbuds", x1, y1)

        # Top-right stack, directly under the FPS counter (FPS is drawn at y≈30).
        h, w = frame.shape[:2]
        right_margin = 12
        y = 58
        for msg in state.active_alerts[:4]:
            color = COLOR_YELLOW if "Spectacle" in msg or "Writing" in msg else COLOR_RED
            if "Earbud" in msg or "Bluetooth" in msg:
                color = COLOR_RED
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            x = max(8, w - text_size[0] - right_margin)
            cv2.rectangle(
                frame,
                (x - 4, y - 16),
                (x + text_size[0] + 4, y + 6),
                (0, 0, 0),
                -1,
            )
            cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 24

    @staticmethod
    def _draw_box_label(frame: np.ndarray, label: str, x1: int, y1: int) -> None:
        """Draw a Phone-style class label just above a detection box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.55, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = max(4, x1)
        ty = max(th + 6, y1 - 8)
        cv2.rectangle(
            frame,
            (tx - 3, ty - th - 4),
            (tx + tw + 3, ty + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, label, (tx, ty), font, scale, COLOR_RED, thickness)
