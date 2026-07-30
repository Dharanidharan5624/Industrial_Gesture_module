"""Bluetooth / earbuds detector — YOLO when available, else enhanced distance- and pose-aware heuristics."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from compliance.detectors.base import BaseDetector
from compliance.models import DetectorResult
from constants import COMPLIANCE_EVENT_BLUETOOTH, COMPLIANCE_EVENT_EARBUDS

try:
    import mediapipe as mp  # type: ignore

    _MP_OK = hasattr(mp, "solutions")
except Exception:
    mp = None
    _MP_OK = False

BBox = Tuple[int, int, int, int]


class EarbudsDetector(BaseDetector):
    """Detect earbuds / Bluetooth headsets worn in ears OR held in palm.

    Priority:
      1. Custom YOLO weights (if provided)
      2. Distance-scaled, pose-aware MediaPipe ear ROI analysis (worn)
      3. Palm / hand-held bud analysis (show-to-camera)
      4. OpenCV face-cascade ear crops (fallback if FaceMesh unavailable)

    Returns DetectorResult with precise detail ("Left Earbud Detected",
    "Bluetooth Device — Right Ear", "Both Earbuds Detected", etc.) and updates
    per-ear bounding boxes (`last_left_bbox`, `last_right_bbox`).
    """

    name = "bluetooth_earbuds"
    event_type = COMPLIANCE_EVENT_EARBUDS

    # MediaPipe Face Mesh ear tragus & pinna landmarks (strictly ears, NO jaw/beard points)
    # Wearer's right ear region (appears on image left in camera feed)
    _MESH_RIGHT = (234, 127, 162, 132)
    # Wearer's left ear region (appears on image right in camera feed)
    _MESH_LEFT = (454, 356, 389, 361)

    _LM_NOSE_TIP = 1
    _LM_CHIN = 152
    _LM_RIGHT_EYE_OUTER = 33   # Wearer's right eye (appears on camera-left)
    _LM_LEFT_EYE_OUTER = 263   # Wearer's left eye (appears on camera-right)

    _CONFIRM_FRAMES = 3
    _CLEAR_FRAMES = 3

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.55,
        weights_path: Optional[str] = None,
    ):
        super().__init__(enabled=enabled, confidence_threshold=confidence_threshold)
        self.weights_path = weights_path
        self._model = None
        self._load_attempted = False
        self.last_kind = "earbuds"
        self.last_side = ""
        self.last_left_bbox: Optional[BBox] = None
        self.last_right_bbox: Optional[BBox] = None

        self._face_mesh = None
        self._hands = None
        self._face_cascade = None

        self._ema_left = 0.0
        self._ema_right = 0.0
        self._streak_left = 0
        self._streak_right = 0
        self._clear_left = 0
        self._clear_right = 0
        self._latched_left = False
        self._latched_right = False

        self._ema_hand = 0.0
        self._streak_hand = 0
        self._clear_hand = 0
        self._latched_hand = False

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
                print(f"[compliance] EarbudsDetector FaceMesh init failed: {exc}")
            try:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as exc:
                print(f"[compliance] EarbudsDetector Hands init failed: {exc}")

        try:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade)
            if self._face_cascade.empty():
                self._face_cascade = None
        except Exception:
            self._face_cascade = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        path = self.weights_path
        if not path or not os.path.isfile(path):
            return False
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(path)
            return True
        except Exception as exc:
            if not self.model_unavailable_logged:
                print(f"[compliance] EarbudsDetector YOLO load failed: {exc}")
                self.model_unavailable_logged = True
            return False

    def detect(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            self._clear_boxes()
            return DetectorResult(detail="disabled")

        if self._ensure_model():
            yolo_result = self._detect_yolo(frame)
            if yolo_result.detected:
                self._latched_hand = False
                return yolo_result

        # 1) Worn-in-ear path (distance-scaled, tragus-anchored & pose-aware)
        ear = self._detect_ear_roi(frame)
        if ear.detected:
            self._latched_hand = False
            self._streak_hand = self._clear_hand = 0
            self._ema_hand = 0.0
            return ear

        # 2) Hand-held path (stem earbud resting on palm)
        hand = self._detect_in_hand(frame, context)
        if hand.detected:
            return hand

        # Nothing solid -> clear & return OK
        self._clear_boxes()
        return DetectorResult(
            detected=False,
            confidence=float(max(ear.confidence, hand.confidence)),
            detail="no_earbuds",
        )

    # ------------------------------------------------------------------ YOLO
    def _detect_yolo(self, frame: np.ndarray) -> DetectorResult:
        try:
            results = self._model.predict(frame, verbose=False, conf=self.confidence_threshold)
        except Exception as exc:
            return DetectorResult(detail=f"infer_error:{exc}")

        best_conf, kind, bbox = 0.0, None, None
        for r in results:
            if r.boxes is None:
                continue
            names = r.names or {}
            for box in r.boxes:
                conf = float(box.conf.item()) if box.conf is not None else 0.0
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                label = str(names.get(cls_id, "")).lower()
                bud_like = any(
                    k in label
                    for k in ("earbud", "earphone", "airpod", "bud", "headset", "bluetooth", "neck")
                )
                if names and not bud_like:
                    bud_like = True
                if not bud_like:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].tolist()
                    bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    kind = (
                        "bluetooth"
                        if any(k in label for k in ("neck", "bluetooth", "headset"))
                        else "earbuds"
                    )

        if kind and bbox:
            side = self._side_from_bbox(frame, bbox)
            self.last_kind = kind
            self.last_side = side
            self.last_left_bbox = bbox if side in ("left", "both", "") else None
            self.last_right_bbox = bbox if side in ("right", "both") else None
            return DetectorResult(
                detected=True,
                confidence=best_conf,
                detail=self._detail_for(kind, side or "left"),
                bbox=bbox,
            )
        return DetectorResult(detected=False, confidence=best_conf, detail="no_earbuds")

    # -------------------------------------------------------- Hand-held path
    def _detect_in_hand(
        self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> DetectorResult:
        """Detect an earbud resting in / held by palm (show-to-camera)."""
        palm = self._palm_bbox(frame, context)
        score, bud_box = 0.0, None
        if palm is not None:
            x1, y1, x2, y2 = palm
            roi = frame[y1:y2, x1:x2]
            score, local = self._score_bud_in_hand(roi)
            if local is not None:
                bx, by, bw, bh = local
                bud_box = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)
            elif score >= 0.40:
                pw, ph = x2 - x1, y2 - y1
                bud_box = (
                    x1 + int(pw * 0.30),
                    y1 + int(ph * 0.25),
                    x1 + int(pw * 0.70),
                    y1 + int(ph * 0.75),
                )

        self._ema_hand = 0.35 * self._ema_hand + 0.65 * score
        thr = max(0.55, float(self.confidence_threshold) + 0.08)
        raw = self._ema_hand >= thr and score >= thr

        if raw:
            self._streak_hand += 1
            self._clear_hand = 0
            if self._streak_hand >= self._CONFIRM_FRAMES:
                self._latched_hand = True
        else:
            self._streak_hand = 0
            self._clear_hand += 1
            if self._clear_hand >= self._CLEAR_FRAMES:
                self._latched_hand = False

        if not self._latched_hand:
            self._clear_boxes()
            self.last_side = ""
            return DetectorResult(
                detected=False,
                confidence=float(self._ema_hand),
                detail="no_earbuds",
            )

        self.last_kind = "earbuds"
        self.last_side = ""
        self.last_left_bbox = bud_box
        self.last_right_bbox = None
        return DetectorResult(
            detected=True,
            confidence=float(min(1.0, max(self._ema_hand, score))),
            detail="Earbuds Detected",
            bbox=bud_box,
        )

    def _palm_bbox(
        self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> Optional[BBox]:
        h, w = frame.shape[:2]
        lms = None
        if context:
            lms = context.get("landmarks")
        if lms is not None:
            box = self._bbox_from_hand_landmarks(lms, w, h)
            if box is not None:
                return box

        if self._hands is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = self._hands.process(rgb)
            if not out.multi_hand_landmarks:
                return None
            best = None
            best_area = 0
            for hand in out.multi_hand_landmarks:
                box = self._bbox_from_hand_landmarks(hand, w, h)
                if box is None:
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > best_area:
                    best_area, best = area, box
            return best
        except Exception:
            return None

    @staticmethod
    def _bbox_from_hand_landmarks(lms, w: int, h: int) -> Optional[BBox]:
        """Palm-focused box from MediaPipe hand landmarks (21 points)."""
        try:
            pts = lms.landmark
        except AttributeError:
            pts = lms
        try:
            xs = [float(p.x) * w for p in pts]
            ys = [float(p.y) * h for p in pts]
        except Exception:
            return None
        if len(xs) < 21:
            if len(xs) < 5:
                return None
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
        else:
            idxs = (0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18)
            pxs = [xs[i] for i in idxs]
            pys = [ys[i] for i in idxs]
            x1, x2 = int(min(pxs)), int(max(pxs))
            y1, y2 = int(min(pys)), int(max(pys))
            tip_xs = [xs[i] for i in (4, 8, 12, 16, 20)]
            tip_ys = [ys[i] for i in (4, 8, 12, 16, 20)]
            x1 = min(x1, int(min(tip_xs)))
            x2 = max(x2, int(max(tip_xs)))
            y1 = min(y1, int(min(tip_ys)))
            y2 = max(y2, int(max(tip_ys)))

        pad = int(0.08 * max(x2 - x1, y2 - y1, 20))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 - x1 < 40 or y2 - y1 < 40:
            return None
        return (x1, y1, x2, y2)

    def _score_bud_in_hand(self, roi: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """Score black/stem bud on palm crop. Ensures solid bud + stem structure."""
        if roi is None or roi.size == 0:
            return 0.0, None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        rh, rw = gray.shape
        area = float(rh * rw)
        if area < 200:
            return 0.0, None

        black = (v_ch < 85) & (s_ch < 100)
        very_black = (v_ch < 55) & (s_ch < 90)
        colorful = (s_ch > 80) & (v_ch > 55) & ((h_ch < 30) | ((h_ch > 12) & (h_ch < 45)))

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        local = cv2.blur(blur.astype(np.float32), (11, 11))
        dark_c = local - blur.astype(np.float32)

        mask = (((black & (dark_c > 8)) | very_black) & (~colorful)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None

        best_score = 0.0
        best_box = None
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            blob = float(cv2.contourArea(c))
            blob_ratio = blob / area
            if blob_ratio < 0.004 or blob_ratio > 0.25:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if min(bw, bh) < 6 or max(bw, bh) < 14:
                continue
            if bw * bh < 120:
                continue
            aspect = max(bw, bh) / float(max(1, min(bw, bh)))
            if aspect > 4.5:
                continue
            peri = float(cv2.arcLength(c, True)) + 1e-6
            circ = 4.0 * np.pi * blob / (peri * peri)

            blob_mask = np.zeros((rh, rw), dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            sat_mean = float(np.mean(s_ch[blob_mask > 0]))
            v_mean = float(np.mean(v_ch[blob_mask > 0]))
            if sat_mean > 80 or v_mean > 110:
                continue

            stem_bonus = 0.0
            for (sy1, sy2, sx1, sx2) in (
                (min(rh, by + bh), min(rh, by + bh + max(10, bh)), max(0, bx - 2), min(rw, bx + bw + 2)),
                (max(0, by - max(10, bh)), by, max(0, bx - 2), min(rw, bx + bw + 2)),
            ):
                if sy2 <= sy1 or sx2 <= sx1:
                    continue
                stem = gray[sy1:sy2, sx1:sx2]
                stem_dark = stem < 90
                sr = float(np.count_nonzero(stem_dark)) / float(stem.size)
                if 0.06 <= sr <= 0.75:
                    col = np.mean(stem_dark.astype(np.float32), axis=0)
                    if float(np.max(col)) > 0.18:
                        stem_bonus = max(stem_bonus, 0.30)

            contrast_mean = float(np.mean(np.maximum(dark_c, 0)[blob_mask > 0]))
            score = (
                0.28 * min(1.0, blob_ratio / 0.04)
                + 0.22 * min(1.0, (1.0 - v_mean / 100.0))
                + 0.12 * min(1.0, max(contrast_mean, 1.0) / 24.0)
                + 0.10 * min(1.0, max(circ, 0.12) / 0.45)
                + stem_bonus
                + (0.08 if sat_mean < 40 else 0.0)
                + (0.08 if 1.2 <= aspect <= 2.8 else 0.0)
            )
            if stem_bonus < 0.25:
                continue
            if score > best_score:
                best_score = score
                best_box = (bx, by, bw, bh)

        return float(min(1.0, best_score)), best_box

    # ----------------------------------------------------------- Ear ROI path
    def _detect_ear_roi(self, frame: np.ndarray) -> DetectorResult:
        hits: List[Tuple[str, float, str, BBox]] = []  # (side, score, kind, box)

        if self._face_mesh is not None:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = self._face_mesh.process(rgb)
            if out.multi_face_landmarks:
                lm = out.multi_face_landmarks[0].landmark
                yaw, pitch, roll, d_eye, nose_x = self._estimate_head_pose(lm, w, h)

                left_occluded = (yaw < -0.35)
                right_occluded = (yaw > 0.35)

                if not left_occluded:
                    s_left, k_left, b_left = self._analyze_ear_landmark(
                        lm, frame, w, h, self._MESH_LEFT, "left", d_eye, yaw
                    )
                    if b_left is not None and s_left >= 0.30:
                        hits.append(("left", s_left, k_left, b_left))

                if not right_occluded:
                    s_right, k_right, b_right = self._analyze_ear_landmark(
                        lm, frame, w, h, self._MESH_RIGHT, "right", d_eye, yaw
                    )
                    if b_right is not None and s_right >= 0.30:
                        hits.append(("right", s_right, k_right, b_right))

                if not hits:
                    s_left2, k_left2, b_left2 = self._analyze_ear_scaled(
                        lm, frame, w, h, self._MESH_LEFT, "left", d_eye, nose_x, yaw
                    )
                    if b_left2 is not None and s_left2 >= 0.30:
                        hits.append(("left", s_left2, k_left2, b_left2))

                    s_right2, k_right2, b_right2 = self._analyze_ear_scaled(
                        lm, frame, w, h, self._MESH_RIGHT, "right", d_eye, nose_x, yaw
                    )
                    if b_right2 is not None and s_right2 >= 0.30:
                        hits.append(("right", s_right2, k_right2, b_right2))

            else:
                self._clear_boxes()
                return DetectorResult(detected=False, confidence=0.0, detail="no_face")
        else:
            if not self.model_unavailable_logged:
                print("[compliance] EarbudsDetector: no FaceMesh — using cascade fallback")
                self.model_unavailable_logged = True
            cascade_hits = self._analyze_ears_cascade(frame)
            if cascade_hits:
                for side, score, box in cascade_hits:
                    if score >= 0.30:
                        hits.append((side, score, "earbuds", box))
            if not hits:
                self._clear_boxes()
                return self.unavailable_result()

        left_score, right_score = 0.0, 0.0
        left_box, right_box = None, None
        left_kind, right_kind = "earbuds", "earbuds"

        for side, score, kind, box in hits:
            if side == "left" and score >= left_score:
                left_score, left_box, left_kind = score, box, kind
            elif side == "right" and score >= right_score:
                right_score, right_box, right_kind = score, box, kind

        if left_box is None and right_box is None:
            self._reset_tracking()
            self._clear_boxes()
            return DetectorResult(detected=False, confidence=0.0, detail="no_face")

        if left_score > 0 and right_score > 0:
            if left_score - right_score > 0.20:
                right_score *= 0.40
            elif right_score - left_score > 0.20:
                left_score *= 0.40

        self._ema_left = 0.40 * self._ema_left + 0.60 * left_score
        self._ema_right = 0.40 * self._ema_right + 0.60 * right_score

        thr = max(0.42, float(self.confidence_threshold) - 0.05)
        left_raw = self._ema_left >= thr
        right_raw = self._ema_right >= thr

        if left_raw and right_raw and min(self._ema_left, self._ema_right) < thr + 0.06:
            if self._ema_left >= self._ema_right:
                right_raw = False
            else:
                left_raw = False

        left_hit = self._update_latch(left_raw, "left")
        right_hit = self._update_latch(right_raw, "right")

        self.last_left_bbox = left_box if left_hit else None
        self.last_right_bbox = right_box if right_hit else None

        if not left_hit and not right_hit:
            self.last_side = ""
            self.last_kind = "earbuds"
            return DetectorResult(
                detected=False,
                confidence=float(max(self._ema_left, self._ema_right)),
                detail="no_earbuds",
            )

        if left_hit and right_hit:
            side, bbox = "both", (left_box if self._ema_left >= self._ema_right else right_box)
            conf = min(self._ema_left, self._ema_right)
            kind = "bluetooth" if (left_kind == "bluetooth" or right_kind == "bluetooth") else "earbuds"
        elif left_hit:
            side, bbox, conf, kind = "left", left_box, self._ema_left, left_kind
        else:
            side, bbox, conf, kind = "right", right_box, self._ema_right, right_kind

        self.last_kind = kind
        self.last_side = side
        return DetectorResult(
            detected=True,
            confidence=float(min(1.0, conf)),
            detail=self._detail_for(kind, side),
            bbox=bbox,
        )

    def _estimate_head_pose(self, lm, w: int, h: int) -> Tuple[float, float, float, float, float]:
        """Estimate 3D head yaw, pitch, roll, inter-eye distance D_eye, and nose_x."""
        nose = (lm[self._LM_NOSE_TIP].x * w, lm[self._LM_NOSE_TIP].y * h)
        chin = (lm[self._LM_CHIN].x * w, lm[self._LM_CHIN].y * h)
        l_eye = (lm[self._LM_LEFT_EYE_OUTER].x * w, lm[self._LM_LEFT_EYE_OUTER].y * h)
        r_eye = (lm[self._LM_RIGHT_EYE_OUTER].x * w, lm[self._LM_RIGHT_EYE_OUTER].y * h)

        mid_eyes = (0.5 * (l_eye[0] + r_eye[0]), 0.5 * (l_eye[1] + r_eye[1]))
        d_eye = math.hypot(r_eye[0] - l_eye[0], r_eye[1] - l_eye[1]) + 1e-5

        # Yaw ratio: offset of nose relative to midpoint of eyes normalized by eye distance
        yaw = (nose[0] - mid_eyes[0]) / d_eye

        # Pitch ratio: vertical distance from eyes to nose vs chin
        pitch = (nose[1] - mid_eyes[1]) / (d_eye + 1e-5) - 0.5

        # Roll angle (radians): tilt between outer eye corners
        roll = math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0])

        return float(yaw), float(pitch), float(roll), float(d_eye), float(nose[0])

    def _analyze_ear_landmark(
        self,
        lm,
        frame: np.ndarray,
        w: int,
        h: int,
        indices: Tuple[int, ...],
        side: str,
        d_eye: float,
        yaw: float,
    ) -> Tuple[float, str, Optional[BBox]]:
        """ROI centered on actual MediaPipe ear landmarks, scaled by inter-eye distance."""
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        ear_cx = int(np.mean(xs))
        ear_cy = int(np.mean(ys))

        box_w = max(28, int(0.50 * d_eye))
        box_h = max(34, int(1.35 * box_w))

        if side == "right":
            ear_cx -= int(0.08 * d_eye)
        else:
            ear_cx += int(0.08 * d_eye)

        ear_cy += int(0.05 * d_eye)

        x1 = max(0, ear_cx - box_w // 2)
        x2 = min(w, ear_cx + box_w // 2)
        y1 = max(0, ear_cy - box_h // 3)
        y2 = min(h, ear_cy + int(box_h * 0.65))

        if x2 - x1 < 10 or y2 - y1 < 14:
            return 0.0, "earbuds", None

        roi = frame[y1:y2, x1:x2]
        score, kind = self._score_bud_in_roi(roi, side)
        return score, kind, (x1, y1, x2, y2)

    def _analyze_ear_scaled(
        self,
        lm,
        frame: np.ndarray,
        w: int,
        h: int,
        indices: Tuple[int, ...],
        side: str,
        d_eye: float,
        nose_x: float,
        yaw: float,
    ) -> Tuple[float, str, Optional[BBox]]:
        """Distance-scaled ear crop analysis using hybrid tragus landmark and eye vector geometry."""
        l_eye = (lm[self._LM_LEFT_EYE_OUTER].x * w, lm[self._LM_LEFT_EYE_OUTER].y * h)
        r_eye = (lm[self._LM_RIGHT_EYE_OUTER].x * w, lm[self._LM_RIGHT_EYE_OUTER].y * h)

        dx = l_eye[0] - r_eye[0]
        dy = l_eye[1] - r_eye[1]
        dist = math.hypot(dx, dy) + 1e-5

        # Unit vector along eye line (from right eye to left eye)
        ux, uy = dx / dist, dy / dist
        # Perpendicular unit vector pointing down towards jaw
        vx, vy = -uy, ux

        # Dynamic ROI dimensions tightly scaled by inter-eye distance (D_eye)
        box_w = max(32, int(0.60 * d_eye))
        box_h = max(38, int(1.30 * box_w))

        if side == "right":  # Wearer's Right Ear (appears on camera-left)
            offset_u = -(0.60 + 0.20 * max(0.0, -yaw))
            ear_cx = int(r_eye[0] + offset_u * d_eye * ux + 0.06 * d_eye * vx)
            ear_cy = int(r_eye[1] + offset_u * d_eye * uy + 0.06 * d_eye * vy)
            # Ensure crop stays strictly to the left of the right outer eye corner
            ear_cx = min(ear_cx, int(r_eye[0] - 8 - box_w // 2))
        else:                # Wearer's Left Ear (appears on camera-right)
            offset_u = +(0.60 + 0.20 * max(0.0, yaw))
            ear_cx = int(l_eye[0] + offset_u * d_eye * ux + 0.06 * d_eye * vx)
            ear_cy = int(l_eye[1] + offset_u * d_eye * uy + 0.06 * d_eye * vy)
            # Ensure crop stays strictly to the right of the left outer eye corner
            ear_cx = max(ear_cx, int(l_eye[0] + 8 + box_w // 2))

        x1 = max(0, ear_cx - box_w // 2)
        x2 = min(w, ear_cx + box_w // 2)
        y1 = max(0, ear_cy - box_h // 3)
        y2 = min(h, ear_cy + int(box_h * 0.65))

        # Safety Check: Reject if crop overlaps the outer eye corners or eyes
        if side == "right" and x2 >= int(r_eye[0]) - 2:
            return 0.0, "earbuds", None
        if side == "left" and x1 <= int(l_eye[0]) + 2:
            return 0.0, "earbuds", None

        if x2 - x1 < 12 or y2 - y1 < 16:
            return 0.0, "earbuds", None

        roi = frame[y1:y2, x1:x2]
        score, kind = self._score_bud_in_roi(roi, side)
        return score, kind, (x1, y1, x2, y2)

    def _update_latch(self, raw_hit: bool, side: str) -> bool:
        if side == "left":
            streak, clear, latched = self._streak_left, self._clear_left, self._latched_left
        else:
            streak, clear, latched = self._streak_right, self._clear_right, self._latched_right

        if raw_hit:
            streak += 1
            clear = 0
            if streak >= self._CONFIRM_FRAMES:
                latched = True
        else:
            streak = 0
            clear += 1
            if clear >= self._CLEAR_FRAMES:
                latched = False

        if side == "left":
            self._streak_left, self._clear_left, self._latched_left = streak, clear, latched
        else:
            self._streak_right, self._clear_right, self._latched_right = streak, clear, latched
        return latched

    def _reset_tracking(self) -> None:
        self._ema_left = self._ema_right = 0.0
        self._streak_left = self._streak_right = 0
        self._clear_left = self._clear_right = 0
        self._latched_left = self._latched_right = False
        self.last_side = ""

    def _clear_boxes(self) -> None:
        self.last_left_bbox = None
        self.last_right_bbox = None

    @staticmethod
    def _anatomical_side(box: BBox, nose_x: float) -> str:
        cx = (box[0] + box[2]) * 0.5
        return "right" if cx < nose_x else "left"

    def _analyze_ears_cascade(self, frame: np.ndarray) -> List[Tuple[str, float, BBox]]:
        if self._face_cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if faces is None or len(faces) == 0:
            return []
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        nose_x = x + fw * 0.5
        ear_w = max(24, int(fw * 0.16))
        ear_h = max(30, int(fh * 0.28))
        ey = y + int(fh * 0.20)
        out: List[Tuple[str, float, BBox]] = []

        rx1, rx2 = max(0, x - ear_w), x + int(fw * 0.05)
        ry1, ry2 = ey, min(frame.shape[0], ey + ear_h)
        rbox = (rx1, ry1, rx2, ry2)
        score_r, _ = self._score_bud_in_roi(frame[ry1:ry2, rx1:rx2], "right")
        out.append(("right", score_r, rbox))

        lx1, lx2 = x + fw - int(fw * 0.05), min(frame.shape[1], x + fw + ear_w)
        ly1, ly2 = ey, min(frame.shape[0], ey + ear_h)
        lbox = (lx1, ly1, lx2, ly2)
        score_l, _ = self._score_bud_in_roi(frame[ly1:ly2, lx1:lx2], "left")
        out.append(("left", score_l, lbox))
        return out

    @staticmethod
    def _skin_ring_ratio(blob_mask: np.ndarray, v_ch: np.ndarray, s_ch: np.ndarray) -> float:
        """Fraction of a ring surrounding candidate blob that matches human skin tone."""
        ring = cv2.dilate(blob_mask, np.ones((5, 5), np.uint8), iterations=2)
        ring = cv2.subtract(ring, blob_mask)
        if not np.any(ring):
            return 0.0
        skin = (v_ch > 65) & (v_ch < 225) & (s_ch > 12) & (s_ch < 145) & (ring > 0)
        return float(np.count_nonzero(skin)) / float(np.count_nonzero(ring))

    def _score_bud_in_roi(self, roi: np.ndarray, side: str = "") -> Tuple[float, str]:
        """Score dark earbuds, white AirPods, and Bluetooth headsets in ear ROI."""
        if roi is None or roi.size == 0:
            return 0.0, "earbuds"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        rh, rw = gray.shape
        area = float(rh * rw)
        if area < 40:
            return 0.0, "earbuds"

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        local = cv2.blur(blur.astype(np.float32), (9, 9))
        dark_c = local - blur.astype(np.float32)
        bright_c = blur.astype(np.float32) - local

        black = (v_ch < 80) & (s_ch < 95)
        very_black = (v_ch < 50) & (s_ch < 85)
        black_ratio = float(np.count_nonzero(black)) / area
        core_ratio = float(np.count_nonzero(very_black)) / area

        colorful = (s_ch > 80) & (v_ch > 55) & ((h_ch < 30) | ((h_ch > 12) & (h_ch < 45)))
        color_ratio = float(np.count_nonzero(colorful)) / area

        black_score = 0.0
        kind_b = "earbuds"

        black_obj = ((black & (dark_c > 8)) | very_black) & (~colorful)
        mask = black_obj.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            ranked = sorted(contours, key=cv2.contourArea, reverse=True)
            best = None
            for c in ranked[:5]:
                blob = float(cv2.contourArea(c))
                blob_ratio = blob / area
                if blob_ratio < 0.012 or blob_ratio > 0.45:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if min(bw, bh) < 3:
                    continue
                aspect = max(bw, bh) / float(max(1, min(bw, bh)))
                if aspect > 5.0:
                    continue
                best = c
                break

            if best is not None:
                blob = float(cv2.contourArea(best))
                blob_ratio = blob / area
                bx, by, bw, bh = cv2.boundingRect(best)
                aspect = max(bw, bh) / float(max(1, min(bw, bh)))
                peri = float(cv2.arcLength(best, True)) + 1e-6
                circ = 4.0 * np.pi * blob / (peri * peri)
                blob_mask = np.zeros((rh, rw), dtype=np.uint8)
                cv2.drawContours(blob_mask, [best], -1, 255, -1)
                sat_mean = float(np.mean(s_ch[blob_mask > 0])) if np.any(blob_mask) else 0.0
                v_mean = float(np.mean(v_ch[blob_mask > 0])) if np.any(blob_mask) else 255.0

                stem_bonus = 0.0
                sy1, sy2 = min(rh, by + bh), min(rh, by + bh + max(8, int(bh * 0.7)))
                sx1, sx2 = max(0, bx - 3), min(rw, bx + bw + 3)
                if sy2 > sy1 and sx2 > sx1:
                    stem = gray[sy1:sy2, sx1:sx2]
                    stem_dark = stem < 90
                    sr = float(np.count_nonzero(stem_dark)) / float(stem.size)
                    if 0.06 <= sr <= 0.75:
                        col = np.mean(stem_dark.astype(np.float32), axis=0)
                        if float(np.max(col)) > 0.18:
                            stem_bonus = 0.30

                if sat_mean <= 75 and v_mean < 110:
                    dark_bonus = 0.18 if v_mean < 55 else (0.10 if v_mean < 80 else 0.0)
                    contrast_mean = float(np.mean(np.maximum(dark_c, 0)[blob_mask > 0])) if np.any(blob_mask) else 0.0
                    black_score = (
                        0.28 * min(1.0, max(core_ratio, black_ratio * 0.30) / 0.035)
                        + 0.24 * min(1.0, blob_ratio / 0.06)
                        + 0.12 * min(1.0, max(contrast_mean, 1.0) / 20.0)
                        + 0.08 * min(1.0, max(circ, 0.08) / 0.35)
                        + stem_bonus
                        + dark_bonus
                        + (0.06 if sat_mean < 35 else 0.0)
                        + (0.06 if 1.0 <= aspect <= 3.0 else 0.0)
                    )
                    if stem_bonus >= 0.15 or core_ratio >= 0.025:
                        black_score = min(1.0, black_score + 0.10)

                    if aspect > 2.0 and blob_ratio > 0.10:
                        kind_b = "bluetooth"

        white_score = 0.0
        kind_w = "earbuds"
        light = (v_ch > 150) & (s_ch < 65) & (~colorful)
        light_obj = light & ((bright_c > 4) | (v_ch > 200))
        if float(np.count_nonzero(light_obj)) / area >= 0.015:
            lmask = light_obj.astype(np.uint8) * 255
            lmask = cv2.morphologyEx(lmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            lcnt, _ = cv2.findContours(lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if lcnt:
                lb = max(lcnt, key=cv2.contourArea)
                lblob = float(cv2.contourArea(lb)) / area
                lx, ly, lw, lh = cv2.boundingRect(lb)
                laspect = max(lw, lh) / float(max(1, min(lw, lh)))
                if 0.018 <= lblob <= 0.40 and laspect <= 4.0:
                    white_score = 0.40 * min(1.0, lblob / 0.05) + 0.34

        score = max(black_score, white_score)
        kind = kind_b if black_score >= white_score else kind_w
        return float(min(1.0, max(0.0, score))), kind

    # --------------------------------------------------------------- Helpers
    @staticmethod
    def _detail_for(kind: str, side: str) -> str:
        if kind == "bluetooth":
            return {
                "left": "Bluetooth Device — Left Ear",
                "right": "Bluetooth Device — Right Ear",
                "both": "Bluetooth Device — Both Ears",
            }.get(side, "Bluetooth Device Detected")
        return {
            "left": "Left Earbud Detected",
            "right": "Right Earbud Detected",
            "both": "Both Earbuds Detected",
        }.get(side, "Earbuds Detected")

    @staticmethod
    def _side_from_bbox(frame: np.ndarray, bbox: BBox) -> str:
        mid = (bbox[0] + bbox[2]) * 0.5
        w = frame.shape[1]
        if mid <= w * 0.45:
            return "right"
        if mid >= w * 0.55:
            return "left"
        return ""
