"""Mobile phone detector — Ultralytics YOLO COCO cell-phone class."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from compliance.detectors.base import BaseDetector
from compliance.models import DetectorResult
from constants import COMPLIANCE_EVENT_MOBILE_PHONE

_COCO_CELL_PHONE = 67
BBox = Tuple[int, int, int, int]

# Absolute fallback next to this package / project root
_HERE = Path(__file__).resolve()
_PROJECT = _HERE.parents[2]  # hagrid-master/


def _resolve_yolo_weights(explicit: Optional[str] = None) -> str:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("COMPLIANCE_PHONE_WEIGHTS", "").strip()
    if env:
        candidates.append(Path(env))
    for root in (_PROJECT, Path.cwd(), _HERE.parents[1]):
        candidates.append(root / "yolov8n.pt")
        candidates.append(root / "weights" / "yolov8n.pt")
        candidates.append(root / "yolov8s.pt")
    for p in candidates:
        try:
            if p.is_file():
                return str(p.resolve())
        except Exception:
            continue
    # Last resort: let Ultralytics download by name
    return "yolov8n.pt"


class PhoneDetector(BaseDetector):
    """Detect mobile phones with YOLO and draw a tight Phone box.

    Trusts COCO cell-phone detections; only blocks full-screen / fist-like FPs.
    """

    name = "mobile_phone"
    event_type = COMPLIANCE_EVENT_MOBILE_PHONE

    # Reduce confirm frames so phone is detected quickly (1 = immediate latch)
    _CONFIRM_FRAMES = 1
    # Keep clear frames low so it de-latches fast when phone leaves frame
    _CLEAR_FRAMES = 3

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.45,
        weights: str = "yolov8n.pt",
    ):
        super().__init__(enabled=enabled, confidence_threshold=confidence_threshold)
        self.weights_path = _resolve_yolo_weights(weights)
        self._model = None
        self._load_attempted = False
        self._ema = 0.0
        self._streak = 0
        self._clear = 0
        self._latched = False
        self.last_bbox: Optional[BBox] = None

    def _ensure_model(self) -> bool:
        if self._model is not None or getattr(self, "_tv_model", None) is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.weights_path)
            print(f"[compliance] PhoneDetector YOLO ready: {self.weights_path}")
            return True
        except Exception as exc:
            print(f"[compliance] PhoneDetector YOLO init failed ({exc}), trying torchvision fallback...")
            try:
                import torch
                from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

                weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                self._tv_model = ssdlite320_mobilenet_v3_large(weights=weights)
                self._tv_model.eval()
                print("[compliance] PhoneDetector torchvision SSDLite MobileNet ready")
                return True
            except Exception as tv_exc:
                if not self.model_unavailable_logged:
                    print(f"[compliance] PhoneDetector model unavailable: {tv_exc}")
                    self.model_unavailable_logged = True
                return False

    def detect(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        if not self.enabled:
            self.last_bbox = None
            return DetectorResult(detail="disabled")
        if frame is None or frame.size == 0:
            return DetectorResult(detail="no_frame")
        if not self._ensure_model():
            self.last_bbox = None
            return self.unavailable_result()

        conf, bbox = self._find_phone(frame)
        hit = bbox is not None and conf >= self._accept_conf()
        return self._finalize(hit, conf, bbox)

    def _accept_conf(self) -> float:
        # Lower accept threshold so live close-up phones at 0.30-0.55 still trigger.
        return max(0.25, min(0.40, float(self.confidence_threshold) - 0.10))

    def _finalize(self, raw_hit: bool, conf: float, bbox: Optional[BBox]) -> DetectorResult:
        self._ema = 0.35 * self._ema + 0.65 * (conf if raw_hit else 0.0)

        if raw_hit and bbox is not None:
            self._streak += 1
            self._clear = 0
            if self._streak >= self._CONFIRM_FRAMES:
                self._latched = True
            self.last_bbox = bbox
        else:
            self._streak = 0
            self._clear += 1
            if self._clear >= self._CLEAR_FRAMES:
                self._latched = False
                self.last_bbox = None

        if not self._latched or self.last_bbox is None:
            return DetectorResult(detected=False, confidence=float(self._ema), detail="no_phone")

        return DetectorResult(
            detected=True,
            confidence=float(min(1.0, max(conf, self._ema))),
            detail="Mobile phone detected",
            bbox=self.last_bbox,
        )

    def _find_phone(self, frame: np.ndarray) -> Tuple[float, Optional[BBox]]:
        h, w = frame.shape[:2]
        cands: List[Tuple[float, BBox]] = []

        def collect(img: np.ndarray, ox: int = 0, oy: int = 0) -> None:
            for conf, box in self._predict_phones(img):
                x1, y1, x2, y2 = box
                mapped = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
                if self._keep_box(mapped, w, h) and self._keep_content(frame, mapped):
                    cands.append((conf, mapped))

        collect(frame)

        # Extra tiles only on HD (720p+) industrial frames — skip for 640px streams
        if max(h, w) >= 900:
            tw, th = max(320, int(w * 0.6)), max(320, int(h * 0.6))
            for ox, oy in (
                (0, 0),
                (max(0, w - tw), 0),
                (0, max(0, h - th)),
                (max(0, w - tw), max(0, h - th)),
            ):
                tile = frame[oy : oy + th, ox : ox + tw]
                if tile.size:
                    collect(tile, ox, oy)

        if not cands:
            return 0.0, None

        # Highest confidence wins; break ties with larger (more complete) phone box
        cands.sort(key=lambda t: (t[0], (t[1][2] - t[1][0]) * (t[1][3] - t[1][1])), reverse=True)
        return cands[0]

    def _predict_phones(self, img: np.ndarray) -> List[Tuple[float, BBox]]:
        conf_gate = max(0.18, self._accept_conf() - 0.08)
        out: List[Tuple[float, BBox]] = []

        if self._model is not None:
            try:
                results = self._model.predict(
                    img,
                    verbose=False,
                    conf=conf_gate,
                    imgsz=640,
                    classes=[_COCO_CELL_PHONE],
                    max_det=8,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls.item()) if box.cls is not None else -1
                        if cls_id != _COCO_CELL_PHONE:
                            continue
                        conf = float(box.conf.item()) if box.conf is not None else 0.0
                        xyxy = box.xyxy[0].tolist()
                        out.append((conf, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))))
            except Exception as exc:
                print(f"[compliance] PhoneDetector predict error: {exc}")

        tv_model = getattr(self, "_tv_model", None)
        if not out and tv_model is not None:
            try:
                import torch
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                with torch.no_grad():
                    preds = tv_model([t])[0]
                boxes = preds["boxes"].cpu().numpy()
                scores = preds["scores"].cpu().numpy()
                labels = preds["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if label == 77 and score >= conf_gate:  # COCO class 77 = cell phone
                        x1, y1, x2, y2 = map(int, box)
                        out.append((float(score), (x1, y1, x2, y2)))
            except Exception as tv_exc:
                print(f"[compliance] PhoneDetector torchvision error: {tv_exc}")

        return out

    @staticmethod
    def _keep_box(bbox: BBox, frame_w: int, frame_h: int) -> bool:
        """Allow real close-up phones; block only absurd full-frame boxes."""
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        area = float(bw * bh)
        frame_area = float(max(1, frame_w * frame_h))
        ratio = area / frame_area
        aspect = max(bw, bh) / float(min(bw, bh))

        # Reject only truly full-screen boxes (laptop background, whole frame)
        if ratio > 0.55:
            return False
        if bw >= 0.92 * frame_w and bh >= 0.92 * frame_h:
            return False
        # Reject tiny noise
        if ratio < 0.001:
            return False
        if min(bw, bh) < 15 or max(bw, bh) < 25:
            return False
        # Accept phone-shaped boxes:
        # - Portrait: tall narrow  (aspect ~1.5 to 6.0)
        # - Landscape: wide short  (aspect ~1.5 to 6.0)
        # - Slight angle: near-square face-on view (aspect ~1.0)
        if aspect < 1.0 or aspect > 7.0:
            return False
        return True

    @staticmethod
    def _keep_content(frame: np.ndarray, bbox: BBox) -> bool:
        """Reject obvious fist-only boxes; keep handsets with screens/bodies."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return False
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hh, s, v = cv2.split(hsv)
        area = float(roi.shape[0] * roi.shape[1])
        skin = ((hh < 28) | (hh > 168)) & (v > 60) & (v < 210) & (s > 30) & (s < 130)
        skin_ratio = float(np.count_nonzero(skin)) / area
        # Raise threshold: only reject if >85% is pure skin (clear fist, no phone visible)
        if skin_ratio > 0.85:
            return False
        return True
