import argparse
import logging
import os
import time
import math
import csv
from collections import Counter, deque
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import mediapipe as mp
from constants import targets
from custom_utils.utils import build_model
from custom_utils.gesture_classifier import classify_gesture
from tool_recognizer import ToolMatcher

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

# Colors (Light Theme)
COLOR_GREEN = (0, 180, 50)
COLOR_RED = (30, 30, 220)
COLOR_YELLOW = (0, 120, 120)        # Darker yellow for text visibility
COLOR_BLUE = (200, 80, 10)
COLOR_ORANGE = (0, 140, 255)        # Orange for in-progress step overlay
COLOR_DARK = (255, 255, 255)        # Canvas background (pure white)
COLOR_LIGHT_DARK = (255, 255, 255)  # Card/Header background (white)
COLOR_WHITE = (255, 255, 255)       # Retained for badge overlays
COLOR_GREY = (120, 120, 120)        # Borders
COLOR_TEXT_MAIN = (50, 50, 50)      # Dark grey text
COLOR_TITLE = (130, 50, 10)         # Deep blue BGR for titles/headings
COLOR_REAL_WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Screen Layout constants
CANVAS_W, CANVAS_H = 1280, 720
CAM_X, CAM_Y, CAM_W, CAM_H = 20, 50, 640, 400

# Step Targets
SCREW_CX, SCREW_CY = 320, 200
SCREW_RADIUS = 25

SCALE_X1, SCALE_Y1, SCALE_X2, SCALE_Y2 = 80, 320, 220, 420
POINT_X1, POINT_Y1, POINT_X2, POINT_Y2 = 420, 80, 560, 180

class RotationTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cumulative_angle = 0.0
        self.last_angle = None

    def update(self, current_angle):
        if self.last_angle is None:
            self.last_angle = current_angle
            return 0.0
        diff = current_angle - self.last_angle
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        if abs(diff) < 0.8:
            self.cumulative_angle += diff
        self.last_angle = current_angle
        return self.cumulative_angle

    def get_turns(self):
        return abs(self.cumulative_angle) / (2 * math.pi)

    def get_direction(self):
        return "Clockwise" if self.cumulative_angle >= 0 else "Anti-Clockwise"

def _bbox_iou(box_a, box_b):
    """Return intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1.0, area_a + area_b - intersection)


def detect_glove(frame, bbox, hand_landmarks=None):
    """Conservatively detect a glove from pixels belonging to the hand.

    The previous implementation measured the complete rectangular box.  That
    box contains a lot of background, and its narrow HSV range also rejected
    darker skin tones, so bare hands were frequently labelled as gloves.
    Here the measurement is restricted to the landmark hull and combines HSV
    and YCrCb skin models.  A low threshold deliberately favours ``Normal``;
    temporal confirmation is applied by :class:`HandDetectionStabilizer`.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return False
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    # Broad ranges support light/dark skin and ordinary indoor illumination.
    hsv_skin = cv2.inRange(hsv, np.array([0, 15, 35]), np.array([28, 230, 255]))
    ycrcb_skin = cv2.inRange(ycrcb, np.array([25, 120, 70]), np.array([255, 190, 150]))
    skin_mask = cv2.bitwise_or(hsv_skin, ycrcb_skin)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    hand_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    if hand_landmarks is not None:
        points = np.array([
            (int(lm.x * w) - x1, int(lm.y * h) - y1)
            for lm in hand_landmarks.landmark
        ], dtype=np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, crop.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, crop.shape[0] - 1)
        cv2.fillConvexPoly(hand_mask, cv2.convexHull(points), 255)

        # Include narrow fingers that can fall outside the palm's convex hull.
        radius = max(2, int(min(crop.shape[:2]) * 0.045))
        for point in points:
            cv2.circle(hand_mask, tuple(point), radius, 255, -1)
    else:
        hand_mask[:] = 255

    hand_pixels = cv2.countNonZero(hand_mask)
    if hand_pixels < 25:
        return False
    skin_ratio = cv2.countNonZero(cv2.bitwise_and(skin_mask, hand_mask)) / hand_pixels
    return skin_ratio < 0.10


class HandDetectionStabilizer:
    """Suppress duplicate MediaPipe hands and stabilize labels/boxes over time."""

    def __init__(self, mirrored=True, history=7):
        self.mirrored = mirrored
        self.history = history
        self.tracks = []
        self.next_track_id = 0

    def handedness_label(self, mediapipe_label):
        # MediaPipe handedness assumes a mirrored/selfie image.  demo_ff flips
        # the frame before inference, therefore its label must NOT be swapped.
        label = mediapipe_label if self.mirrored else ("Right" if mediapipe_label == "Left" else "Left")
        return f"{label} Hand"

    @staticmethod
    def _center_distance(box_a, box_b):
        ac = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
        bc = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
        return math.hypot(ac[0] - bc[0], ac[1] - bc[1])

    def _deduplicate(self, detections):
        kept = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            duplicate = False
            for existing in kept:
                iou = _bbox_iou(detection["bbox"], existing["bbox"])
                min_size = max(1, min(
                    detection["bbox"][2] - detection["bbox"][0],
                    detection["bbox"][3] - detection["bbox"][1],
                    existing["bbox"][2] - existing["bbox"][0],
                    existing["bbox"][3] - existing["bbox"][1],
                ))
                close_centers = self._center_distance(detection["bbox"], existing["bbox"]) < 0.35 * min_size
                if iou >= 0.45 or (iou >= 0.20 and close_centers):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept

    def update(self, detections):
        detections = self._deduplicate(detections)
        for track in self.tracks:
            track["matched"] = False

        stable = []
        for detection in detections:
            candidates = [
                track for track in self.tracks
                if not track["matched"] and (
                    _bbox_iou(track["bbox"], detection["bbox"]) > 0.08
                    or self._center_distance(track["bbox"], detection["bbox"]) < 80
                )
            ]
            track = max(candidates, key=lambda item: _bbox_iou(item["bbox"], detection["bbox"]), default=None)
            if track is None:
                track = {
                    "id": self.next_track_id,
                    "bbox": detection["bbox"],
                    "side_votes": deque(maxlen=self.history),
                    "glove_votes": deque(maxlen=self.history),
                    "missed": 0,
                }
                self.next_track_id += 1
                self.tracks.append(track)

            track["matched"] = True
            track["missed"] = 0
            track["side_votes"].append(detection["side"])
            track["glove_votes"].append(bool(detection["glove_candidate"]))
            # Exponential smoothing removes box jitter without adding latency.
            track["bbox"] = tuple(int(0.65 * old + 0.35 * new) for old, new in zip(track["bbox"], detection["bbox"]))

            side = Counter(track["side_votes"]).most_common(1)[0][0]
            # Require strong multi-frame evidence before calling a glove.
            glove = len(track["glove_votes"]) >= 5 and sum(track["glove_votes"]) >= 5
            stable.append({**detection, "bbox": track["bbox"], "side": side, "glove": glove})

        for track in self.tracks:
            if not track["matched"]:
                track["missed"] += 1
        self.tracks = [track for track in self.tracks if track["missed"] <= 8]
        return stable

class LogBook:
    def __init__(self):
        self.logs = [
            "System initialized. Ready for SOP monitoring.",
            "Employee EMP001 logged in successfully.",
            "Connected to MES endpoint: STATION-3."
        ]

    def add(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 8:
            self.logs.pop(0)

global_scroll_pos = 0
global_is_dragging = False

# Log panel scrollbar state
log_scroll_pos = 0
log_is_dragging = False
# Log scrollbar & box geometry (updated each frame from the render loop)
_LOG_SB_X  = 0
_LOG_SB_Y1 = 0
_LOG_SB_H  = 1
_LOG_BOX_X1 = 0   # left edge of the log box (for wheel-over detection)
_LOG_BOX_Y1 = 0   # top edge
_LOG_BOX_X2 = 0   # right edge
_LOG_BOX_Y2 = 0   # bottom edge
_LOG_MAX_SCROLL = 0  # updated each frame so wheel can clamp correctly

def on_mouse(event, x, y, flags, param):
    global global_scroll_pos, global_is_dragging
    global log_scroll_pos, log_is_dragging

    # ── Reference-frame horizontal slider (bottom of left panel) ──────────
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 640 and 690 <= y <= 730:
            global_is_dragging = True
    elif event == cv2.EVENT_LBUTTONUP:
        global_is_dragging = False
        log_is_dragging = False
    if global_is_dragging and (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN):
        clamped_x = max(20, min(x, 640))
        ratio     = (clamped_x - 20) / 620.0
        global_scroll_pos = int(round(ratio * 4))

    # ── Log panel: scrollbar thumb drag ───────────────────────────────────
    sb_x1 = _LOG_SB_X - 10
    sb_x2 = _LOG_SB_X + 10
    sb_y1 = _LOG_SB_Y1
    sb_y2 = _LOG_SB_Y1 + _LOG_SB_H
    if event == cv2.EVENT_LBUTTONDOWN:
        if sb_x1 <= x <= sb_x2 and sb_y1 <= y <= sb_y2:
            log_is_dragging = True
    if log_is_dragging and (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN):
        clamped_y = max(sb_y1, min(y, sb_y2))
        ratio     = (clamped_y - sb_y1) / max(1, float(_LOG_SB_H))
        log_scroll_pos = int(round(ratio * _LOG_MAX_SCROLL))

    # ── Log panel: mouse-wheel scroll (when cursor is anywhere over the box) ─
    if event == cv2.EVENT_MOUSEWHEEL:
        if _LOG_BOX_X1 <= x <= _LOG_BOX_X2 and _LOG_BOX_Y1 <= y <= _LOG_BOX_Y2:
            # flags > 0  → scroll up (wheel forward) → show earlier logs
            # flags < 0  → scroll down               → show later logs
            if flags > 0:
                log_scroll_pos = max(0, log_scroll_pos - 1)
            else:
                log_scroll_pos = min(_LOG_MAX_SCROLL, log_scroll_pos + 1)


class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        transformed_image = transform(image=img)
        return transformed_image["image"]

    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(classifier, transform, conf: DictConfig) -> None:
        global global_scroll_pos
        global log_scroll_pos
        # Initialize MediaPipe Hands
        hands = None
        mp_hands = None
        if hasattr(mp, "solutions"):
            try:
                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(
                    model_complexity=1,
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.65,
                    min_tracking_confidence=0.65,
                )
            except Exception as e:
                logging.warning(f"Could not initialize MediaPipe Hands: {e}")

        # Try loading reference images
        _HERE = os.path.dirname(os.path.abspath(__file__))
        ref_path = os.path.join(_HERE, "images", "example.jpeg")
        ref_img = cv2.imread(ref_path)
        ref_tiles = []
        if ref_img is not None:
            # Crop 4 simple segments to act as reference frames
            th, tw = ref_img.shape[:2]
            block_w, block_h = tw // 2, th // 2
            ref_tiles.append(cv2.resize(ref_img[0:block_h, 0:block_w], (120, 60)))
            ref_tiles.append(cv2.resize(ref_img[0:block_h, block_w:tw], (120, 60)))
            ref_tiles.append(cv2.resize(ref_img[block_h:th, 0:block_w], (120, 60)))
            ref_tiles.append(cv2.resize(ref_img[block_h:th, block_w:tw], (120, 60)))

        # Load custom Poppins fonts if available
        _HERE = os.path.dirname(os.path.abspath(__file__))
        poppins_reg = os.path.join(_HERE, "Poppins-Regular.ttf")
        poppins_bold = os.path.join(_HERE, "Poppins-Bold.ttf")
        
        font_path = poppins_reg if os.path.exists(poppins_reg) else "C:\\Windows\\Fonts\\arial.ttf"
        font_bold_path = poppins_bold if os.path.exists(poppins_bold) else "C:\\Windows\\Fonts\\arial.ttf"
        
        try:
            font_title = ImageFont.truetype(font_bold_path, 18)
            font_header = ImageFont.truetype(font_bold_path, 15)
            font_body = ImageFont.truetype(font_path, 13)
            font_log = ImageFont.truetype(font_path, 12)
            font_stats_lbl = ImageFont.truetype(font_path, 12)
            font_stats_val = ImageFont.truetype(font_bold_path, 26)
        except Exception:
            font_title = font_header = font_body = font_log = font_stats_lbl = font_stats_val = ImageFont.load_default()

        if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            logging.error("No display session found. Run this demo from a desktop terminal, not a headless shell.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Could not open camera index 0. Connect/enable a webcam, or change cv2.VideoCapture(0).")
            return

        cv2.namedWindow("Industrial Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Industrial Monitor", CANVAS_W, CANVAS_H)
        cv2.setMouseCallback("Industrial Monitor", on_mouse)
        
        # State indicators
        tracker = RotationTracker()
        hand_stabilizer = HandDetectionStabilizer(mirrored=True)
        log_book = LogBook()
        tool_matcher = ToolMatcher(images_dir=os.path.join(_HERE, "images"))
        
        # Statistics
        det_count = 0
        ok_count = 0
        ng_count = 0

        
        # SOP steps
        active_step = 1  # 1: Height, 2: Weight, 3: O-Ring, 4: Pen, 5: Screw, 6: Rotate, 7: Pack, 8: Complete
        step_progress = [0] * 8  # percentage progress of each step
        
        weight_timer = 0.0
        point_timer = 0.0
        
        summary_card_visible = False
        summary_action = ""
        summary_hand = ""
        summary_direction = ""
        summary_turns = 0.0
        dynamic_screw_cx = SCREW_CX
        dynamic_screw_cy = SCREW_CY

        t1 = cnt = 0

        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Mirror correction
            frame = cv2.flip(frame, 1)
            raw_cam_frame = cv2.resize(frame, (CAM_W, CAM_H))
            display_cam_frame = raw_cam_frame.copy()

            # Preprocess and run classification model
            processed_frame = Demo.preprocess(frame, transform).unsqueeze(0)
            model_device = next(classifier.parameters()).device
            processed_frame = processed_frame.to(model_device)
            with torch.no_grad():
                logits = classifier(processed_frame)

            probs = torch.softmax(logits, dim=1)
            nn_conf = float(probs[0].max())
            nn_label_idx = int(probs[0].argmax())
            nn_label_text = targets.get(nn_label_idx, "unknown")

            # Detect hands and bounding boxes using MediaPipe
            bbox_list = []
            rule_gestures = []
            hand_labels = []
            detected_hands = []
            results = None

            if hands is not None:
                results = hands.process(cv2.cvtColor(raw_cam_frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks and results.multi_handedness:
                    raw_detections = []
                    for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Bounding box
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        x1, x2 = int(min(x_coords) * CAM_W), int(max(x_coords) * CAM_W)
                        y1, y2 = int(min(y_coords) * CAM_H), int(max(y_coords) * CAM_H)

                        pad_w = int((x2 - x1) * 0.15)
                        pad_h = int((y2 - y1) * 0.15)
                        x1 = max(0, x1 - pad_w)
                        y1 = max(0, y1 - pad_h)
                        x2 = min(CAM_W, x2 + pad_w)
                        y2 = min(CAM_H, y2 + pad_h)

                        classification = handedness_info.classification[0]
                        raw_detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "landmarks": hand_landmarks,
                            "gesture": classify_gesture(hand_landmarks),
                            "side": hand_stabilizer.handedness_label(classification.label),
                            "confidence": float(classification.score),
                            "glove_candidate": detect_glove(
                                raw_cam_frame, (x1, y1, x2, y2), hand_landmarks
                            ),
                        })

                    detected_hands = hand_stabilizer.update(raw_detections)
                else:
                    hand_stabilizer.update([])

            # All downstream logic and drawing consume the same deduplicated,
            # temporally stable list, guaranteeing one result per physical hand.
            for detection in detected_hands:
                bbox_list.append(detection["bbox"])
                rule_gestures.append(detection["gesture"])
                glove_status = "Glove" if detection["glove"] else "Normal"
                hand_labels.append(f'{detection["side"]} ({glove_status})')

            # Reset indicators
            screw_aligned = False
            scale_aligned = False
            point_aligned = False

            # Dynamic screw and spanner/tool detection
            detected_screw = None
            screw_bbox = None
            detected_tool = None
            tool_bbox = None

            # 1. Screw Detection (Hough Circles & Contour Circularity)
            gray_img = cv2.cvtColor(raw_cam_frame, cv2.COLOR_BGR2GRAY)
            blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred_img,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=100,
                param2=30,
                minRadius=8,
                maxRadius=35
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    if 20 < cx < CAM_W - 20 and 20 < cy < CAM_H - 20:
                        detected_screw = (int(cx), int(cy))
                        screw_bbox = (int(cx - r - 5), int(cy - r - 5), int(cx + r + 5), int(cy + r + 5))
                        break
            
            if detected_screw is None:
                _, thresh = cv2.threshold(blurred_img, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 150 < area < 3000:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * math.pi * area / (perimeter * perimeter)
                            if circularity > 0.75:
                                (x, y, w, h) = cv2.boundingRect(cnt)
                                aspect_ratio = float(w) / h
                                if 0.8 <= aspect_ratio <= 1.2:
                                    cx = x + w // 2
                                    cy = y + h // 2
                                    if 20 < cx < CAM_W - 20 and 20 < cy < CAM_H - 20:
                                        detected_screw = (cx, cy)
                                        screw_bbox = (x, y, x + w, y + h)
                                        break
            
            if detected_screw is not None:
                dynamic_screw_cx = CAM_W - 50
            
            recognized_tool_name = None
            dynamic_screw_cy = detected_screw[1] if detected_screw else dynamic_screw_cy
            if detected_screw: dynamic_screw_cx = detected_screw[0]

            # 2. Tool (Spanner/Screwdriver) Detection near hand
            first_hand_landmarks = detected_hands[0]["landmarks"] if detected_hands else None
            if first_hand_landmarks is not None:
                wrist_x = int(first_hand_landmarks.landmark[0].x * CAM_W)
                wrist_y = int(first_hand_landmarks.landmark[0].y * CAM_H)
                tip_x = int(first_hand_landmarks.landmark[8].x * CAM_W)
                tip_y = int(first_hand_landmarks.landmark[8].y * CAM_H)
                
                # Default fallback (finger extension)
                index_mcp = first_hand_landmarks.landmark[5]
                mcp_x = int(index_mcp.x * CAM_W)
                mcp_y = int(index_mcp.y * CAM_H)
                fallback_tx = tip_x + int((tip_x - mcp_x) * 0.6)
                fallback_ty = tip_y + int((tip_y - mcp_y) * 0.6)
                
                hx1 = max(0, min(wrist_x, tip_x) - 100)
                hy1 = max(0, min(wrist_y, tip_y) - 100)
                hx2 = min(CAM_W, max(wrist_x, tip_x) + 100)
                hy2 = min(CAM_H, max(wrist_y, tip_y) + 100)
                
                hand_roi = raw_cam_frame[hy1:hy2, hx1:hx2]
                if hand_roi.size > 0:
                    roi_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                    roi_edges = cv2.Canny(roi_blurred, 50, 150)
                    contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    best_cnt = None
                    max_len = 0
                    for cnt in contours:
                        rect = cv2.minAreaRect(cnt)
                        (cx, cy), (w, h), angle = rect
                        length = max(w, h)
                        width = min(w, h)
                        if length > 40 and width > 0:
                            aspect_ratio = length / width
                            if aspect_ratio > 2.0:
                                if length > max_len:
                                    max_len = length
                                    best_cnt = cnt
                                    
                    if best_cnt is not None:
                        tx, ty, tw, th = cv2.boundingRect(best_cnt)
                        gtx1, gty1 = tx + hx1, ty + hy1
                        gtx2, gty2 = gtx1 + tw, gty1 + th
                        tool_bbox = (gtx1, gty1, gtx2, gty2)
                        
                        pts = best_cnt.reshape(-1, 2)
                        pts[:, 0] += hx1
                        pts[:, 1] += hy1
                        dists = np.sum((pts - np.array([wrist_x, wrist_y]))**2, axis=1)
                        tip_idx = np.argmax(dists)
                        detected_tool = (int(pts[tip_idx][0]), int(pts[tip_idx][1]))
                        
                        tool_roi = raw_cam_frame[max(0, gty1):min(CAM_H, gty2), max(0, gtx1):min(CAM_W, gtx2)]
                        match_name, _ = tool_matcher.match(tool_roi)
                        if match_name:
                            recognized_tool_name = match_name

            # Draw screw detection box on display frame if found
            if screw_bbox is not None:
                cv2.rectangle(display_cam_frame, (screw_bbox[0], screw_bbox[1]), (screw_bbox[2], screw_bbox[3]), (255, 255, 0), 2)
                cv2.putText(display_cam_frame, "SCREW DETECTED", (screw_bbox[0], screw_bbox[1] - 5), FONT, 0.4, (255, 255, 0), 1)

            # Draw virtual targets based on active SOP step
            if active_step == 5:
                # 5. Draw Screw Target
                cv2.rectangle(display_cam_frame, (dynamic_screw_cx - 35, dynamic_screw_cy - 35), (dynamic_screw_cx + 35, dynamic_screw_cy + 35), COLOR_YELLOW, 2)
                cv2.circle(display_cam_frame, (dynamic_screw_cx, dynamic_screw_cy), 6, COLOR_RED, -1)
                cv2.circle(display_cam_frame, (dynamic_screw_cx, dynamic_screw_cy), SCREW_RADIUS, COLOR_YELLOW, 1)
                cv2.putText(display_cam_frame, "SCREW TARGET", (dynamic_screw_cx - 40, dynamic_screw_cy + 50), FONT, 0.45, COLOR_YELLOW, 1)

            elif active_step == 2:
                # 2. Draw Weight Scale Box
                cv2.rectangle(display_cam_frame, (SCALE_X1, SCALE_Y1), (SCALE_X2, SCALE_Y2), COLOR_BLUE, 2)
                cv2.putText(display_cam_frame, "WEIGH SCALE", (SCALE_X1 + 10, SCALE_Y1 - 10), FONT, 0.45, COLOR_BLUE, 1)

            elif active_step == 4:
                # 4. Draw Point Target Box
                cv2.rectangle(display_cam_frame, (POINT_X1, POINT_Y1), (POINT_X2, POINT_Y2), COLOR_RED, 2)
                cv2.putText(display_cam_frame, "PART TARGET", (POINT_X1 + 5, POINT_Y1 - 10), FONT, 0.45, COLOR_RED, 1)

            # Analyze hand position relative to the active target
            if detected_hands:
                hand_landmarks = detected_hands[0]["landmarks"]
                hand_side = hand_labels[0]
                
                index_mcp = hand_landmarks.landmark[5]
                index_tip = hand_landmarks.landmark[8]
                tip_x = int(index_tip.x * CAM_W)
                tip_y = int(index_tip.y * CAM_H)
                mcp_x = int(index_mcp.x * CAM_W)
                mcp_y = int(index_mcp.y * CAM_H)
                
                if detected_tool is not None:
                    tool_x, tool_y = detected_tool
                else:
                    tool_x = tip_x + int((tip_x - mcp_x) * 0.5)
                    tool_y = tip_y + int((tip_y - mcp_y) * 0.5)

                if active_step == 5:
                    # Screwdriver tool alignment
                    dist = math.sqrt((tool_x - dynamic_screw_cx) ** 2 + (tool_y - dynamic_screw_cy) ** 2)
                    # Project screwdriver/spanner graphic
                    if tool_bbox is not None:
                        color = COLOR_GREEN if recognized_tool_name else COLOR_ORANGE
                        label = f"DETECTED: {recognized_tool_name.upper()}" if recognized_tool_name else "UNKNOWN TOOL"
                        cv2.rectangle(display_cam_frame, (tool_bbox[0], tool_bbox[1]), (tool_bbox[2], tool_bbox[3]), color, 2)
                        cv2.putText(display_cam_frame, label, (tool_bbox[0], tool_bbox[1] - 5), FONT, 0.4, color, 1)
                    else:
                        cv2.rectangle(display_cam_frame, (min(tip_x, tool_x) - 15, min(tip_y, tool_y) - 15),
                                      (max(tip_x, tool_x) + 15, max(tip_y, tool_y) + 15), COLOR_GREEN, 2)
                        cv2.line(display_cam_frame, (tip_x, tip_y), (tool_x, tool_y), COLOR_GREEN, 2)
                        cv2.putText(display_cam_frame, "SCREWDRIVER", (min(tip_x, tool_x), min(tip_y, tool_y) - 10), FONT, 0.4, COLOR_GREEN, 1)

                    if dist < 50:
                        screw_aligned = True
                        cv2.line(display_cam_frame, (tool_x, tool_y), (dynamic_screw_cx, dynamic_screw_cy), COLOR_GREEN, 1)
                        
                        if recognized_tool_name:
                            wrist = hand_landmarks.landmark[0]
                            middle_mcp = hand_landmarks.landmark[9]
                            angle = math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)
                            
                            if not summary_card_visible:
                                tracker.update(angle)
                                turns = tracker.get_turns()
                                step_progress[4] = int(min(100, (turns / 2.5) * 100))
                                if turns >= 2.5:
                                    # Completed Screw Tight
                                    summary_action = "Screw Tight"
                                    summary_hand = hand_side
                                    summary_direction = tracker.get_direction()
                                    summary_turns = turns
                                    summary_card_visible = True
                                    log_book.add(f"{recognized_tool_name.upper()} tightening completed (2.5 turns).")
                        else:
                            cv2.putText(display_cam_frame, "USE CORRECT TOOL", (dynamic_screw_cx - 60, dynamic_screw_cy - 45), FONT, 0.5, COLOR_RED, 2)

                elif active_step == 2:
                    # Check if hand wrist is inside the scale area
                    wrist_x = int(hand_landmarks.landmark[0].x * CAM_W)
                    wrist_y = int(hand_landmarks.landmark[0].y * CAM_H)
                    if SCALE_X1 <= wrist_x <= SCALE_X2 and SCALE_Y1 <= wrist_y <= SCALE_Y2:
                        scale_aligned = True
                        weight_timer += delta
                        step_progress[1] = int(min(100, (weight_timer / 1.5) * 100))
                        
                        # Draw weight readout
                        curr_weight = int(min(450, (weight_timer / 1.5) * 450))
                        cv2.putText(display_cam_frame, f"WEIGHT: {curr_weight}g", (SCALE_X1 + 10, SCALE_Y1 + 45), FONT, 0.5, COLOR_GREEN, 2)
                        
                        if weight_timer >= 1.5:
                            summary_action = "Weight Check"
                            summary_hand = hand_side
                            summary_direction = "N/A"
                            summary_turns = 0.0
                            summary_card_visible = True
                            log_book.add("Weight Check completed (450g OK).")
                    else:
                        weight_timer = max(0.0, weight_timer - delta)

                elif active_step == 4:
                    # Index tip pointing inside target box
                    if POINT_X1 <= tip_x <= POINT_X2 and POINT_Y1 <= tip_y <= POINT_Y2 and rule_gestures[0] == "one":
                        point_aligned = True
                        point_timer += delta
                        step_progress[3] = int(min(100, (point_timer / 1.0) * 100))
                        
                        if point_timer >= 1.0:
                            summary_action = "Pen Pointing"
                            summary_hand = hand_side
                            summary_direction = "N/A"
                            summary_turns = 0.0
                            summary_card_visible = True
                            log_book.add("Pen Pointing inspection completed.")
                    else:
                        point_timer = max(0.0, point_timer - delta)

            # Draw hand boxes & sign classifications
            if bbox_list:
                for idx, bbox in enumerate(bbox_list):
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(display_cam_frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

                    # Display Hand Label (Left/Right)
                    cv2.rectangle(display_cam_frame, (x1, y1 - 42), (x1 + 220, y1), (128, 0, 64), -1)
                    cv2.putText(display_cam_frame, hand_labels[idx], (x1 + 4, y1 - 28), FONT, 0.42, COLOR_WHITE, 1)

                    # Display Neural Network (full-frame) prediction
                    nn_info = f"NN: {nn_label_text} ({nn_conf:.2f})"
                    cv2.putText(display_cam_frame, nn_info, (x1 + 4, y1 - 16), FONT, 0.42, COLOR_WHITE, 1)
                    
                    # Display Rule-based (landmark) prediction
                    rule_info = f"Rule: {rule_gestures[idx]}"
                    cv2.putText(display_cam_frame, rule_info, (x1 + 4, y1 - 4), FONT, 0.42, COLOR_WHITE, 1)

            # Main dashboard canvas composition
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            canvas[:] = COLOR_DARK

            # Title Header Bar
            cv2.rectangle(canvas, (0, 0), (CANVAS_W, 40), COLOR_LIGHT_DARK, -1)

            # Composite the live camera view
            canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = display_cam_frame
            # Camera border
            cv2.rectangle(canvas, (CAM_X, CAM_Y), (CAM_X+CAM_W, CAM_Y+CAM_H), COLOR_BLUE, 2)

            # SOP Progress Bar below camera
            prog_y = CAM_Y + CAM_H + 15
            cv2.rectangle(canvas, (CAM_X, prog_y), (CAM_X + CAM_W, prog_y + 35), COLOR_LIGHT_DARK, -1)
            cv2.rectangle(canvas, (CAM_X, prog_y), (CAM_X + CAM_W, prog_y + 35), COLOR_GREY, 1)

            # Compute total progress
            tot_prog = sum(step_progress) // 8
            bar_w = int((CAM_W - 10) * (tot_prog / 100))
            cv2.rectangle(canvas, (CAM_X + 5, prog_y + 5), (CAM_X + 5 + bar_w, prog_y + 30), COLOR_GREEN, -1)

            # Reference Frames at Bottom
            ref_y = prog_y + 70
            step_names = [
                "1. Height Chk", "2. Weight Chk", "3. O-Ring Chk", "4. Pen Point",
                "5. Screw Align", "6. Rotate Chk", "7. Pack Prod", "8. Complete"
            ]
            scroll_pos = global_scroll_pos

            for display_idx in range(4):
                i = scroll_pos + display_idx
                if i >= 8: continue
                rx = CAM_X + display_idx * 160

                # 1. Determine crop region from display_cam_frame
                if i == 4: # 5. Screw Align
                    cx, cy = dynamic_screw_cx, dynamic_screw_cy
                    x1, x2 = cx - 100, cx + 100
                    y1, y2 = cy - 50, cy + 50
                elif i == 1: # 2. Weight Scale
                    cx, cy = 150, 370
                    x1, x2 = cx - 120, cx + 120
                    y1, y2 = cy - 60, cy + 60
                elif i == 3: # 4. Pen Point
                    cx, cy = 490, 130
                    x1, x2 = cx - 120, cx + 120
                    y1, y2 = cy - 60, cy + 60
                else:
                    cx, cy = 320, 240
                    x1, x2 = cx - 240, cx + 240
                    y1, y2 = cy - 120, cy + 120

                x1 = max(0, x1); x2 = min(CAM_W, x2)
                y1 = max(0, y1); y2 = min(CAM_H, y2)

                crop_img = display_cam_frame[y1:y2, x1:x2].copy()
                tile_img = cv2.resize(crop_img, (120, 60))

                # Status overlay color
                status_color = None
                if active_step == (i + 1):
                    if step_progress[i] == 0:    status_color = COLOR_RED
                    elif step_progress[i] < 100: status_color = COLOR_ORANGE
                    else:                         status_color = COLOR_GREEN
                elif active_step > (i + 1): status_color = COLOR_GREEN
                else:                        status_color = COLOR_RED

                if status_color is not None:
                    overlay = np.zeros_like(tile_img)
                    overlay[:] = status_color
                    tile_img = cv2.addWeighted(overlay, 0.45, tile_img, 0.55, 0)

                cv2.rectangle(canvas, (rx, ref_y), (rx + 140, ref_y + 90), COLOR_LIGHT_DARK, -1)
                border_color = COLOR_GREEN if active_step == (i + 1) else COLOR_GREY
                border_w = 2 if active_step == (i + 1) else 1
                cv2.rectangle(canvas, (rx, ref_y), (rx + 140, ref_y + 90), border_color, border_w)
                canvas[ref_y+5:ref_y+65, rx+10:rx+130] = tile_img

            # --- DRAW CUSTOM SLIDER ---
            slider_y = ref_y + 98
            thumb_x = CAM_X + int((scroll_pos / 4.0) * 620)
            cv2.line(canvas, (CAM_X, slider_y), (CAM_X + 620, slider_y), (150, 60, 0), 6)
            cv2.line(canvas, (CAM_X, slider_y), (thumb_x, slider_y), (255, 120, 0), 6)
            cv2.circle(canvas, (thumb_x, slider_y), 10, (255, 255, 255), -1)
            cv2.circle(canvas, (thumb_x, slider_y), 10, (200, 200, 200), 1)

            # --- RIGHT PANEL ---
            rx_start = CAM_X + CAM_W + 20

            # 1. Statistics Cards
            cv2.rectangle(canvas, (rx_start, CAM_Y), (CANVAS_W - 20, CAM_Y + 95), COLOR_LIGHT_DARK, -1)
            cv2.rectangle(canvas, (rx_start, CAM_Y), (CANVAS_W - 20, CAM_Y + 95), COLOR_GREY, 1)
            for i in range(4):
                cx = rx_start + i * 145
                if i < 3:
                    cv2.line(canvas, (cx + 135, CAM_Y + 15), (cx + 135, CAM_Y + 80), COLOR_GREY, 1)

            # 2. Video Source Control
            vy = CAM_Y + 115
            cv2.rectangle(canvas, (rx_start, vy), (CANVAS_W - 20, vy + 95), COLOR_LIGHT_DARK, -1)
            cv2.rectangle(canvas, (rx_start, vy), (CANVAS_W - 20, vy + 95), COLOR_GREY, 1)

            # 3. MES Integration Box
            my = vy + 115
            cv2.rectangle(canvas, (rx_start, my), (CANVAS_W - 20, my + 95), COLOR_LIGHT_DARK, -1)
            cv2.rectangle(canvas, (rx_start, my), (CANVAS_W - 20, my + 95), COLOR_GREY, 1)
            cv2.rectangle(canvas, (rx_start + 15, my + 42), (rx_start + 120, my + 65), COLOR_GREEN, -1)

            # 4. Employee Login Box
            ey = my + 115
            cv2.rectangle(canvas, (rx_start, ey), (CANVAS_W - 20, ey + 95), COLOR_LIGHT_DARK, -1)
            cv2.rectangle(canvas, (rx_start, ey), (CANVAS_W - 20, ey + 95), COLOR_GREY, 1)

            # 5. Real-Time Action Log
            ly = ey + 115
            LOG_BOX_H = CANVAS_H - ly - 10   # dynamic height: fills to bottom
            cv2.rectangle(canvas, (rx_start, ly), (CANVAS_W - 20, ly + LOG_BOX_H), COLOR_LIGHT_DARK, -1)
            cv2.rectangle(canvas, (rx_start, ly), (CANVAS_W - 20, ly + LOG_BOX_H), COLOR_GREY, 1)

            # --- Log scrollbar geometry ---
            LOG_LINE_H    = 16
            LOG_VISIBLE   = max(1, (LOG_BOX_H - 32) // LOG_LINE_H)  # lines that fit
            total_logs    = len(log_book.logs)
            max_scroll    = max(0, total_logs - LOG_VISIBLE)
            log_scroll_pos = max(0, min(log_scroll_pos, max_scroll))

            # Update global geometry for mouse callback
            global _LOG_SB_X, _LOG_SB_Y1, _LOG_SB_H
            global _LOG_BOX_X1, _LOG_BOX_Y1, _LOG_BOX_X2, _LOG_BOX_Y2, _LOG_MAX_SCROLL
            _LOG_SB_X      = CANVAS_W - 22
            _LOG_SB_Y1     = ly + 30
            _LOG_SB_H      = LOG_BOX_H - 35
            _LOG_BOX_X1    = rx_start        # left edge of the log box
            _LOG_BOX_Y1    = ly              # top edge
            _LOG_BOX_X2    = CANVAS_W - 20  # right edge
            _LOG_BOX_Y2    = ly + LOG_BOX_H # bottom edge
            _LOG_MAX_SCROLL = max_scroll     # for wheel clamping in on_mouse

            # Draw scrollbar track
            cv2.line(canvas, (_LOG_SB_X, _LOG_SB_Y1), (_LOG_SB_X, _LOG_SB_Y1 + _LOG_SB_H), (180, 180, 180), 3)
            # Thumb position
            if max_scroll > 0:
                thumb_ratio = log_scroll_pos / max_scroll
            else:
                thumb_ratio = 0.0
            thumb_y = _LOG_SB_Y1 + int(thumb_ratio * (_LOG_SB_H - 14))
            cv2.rectangle(canvas, (_LOG_SB_X - 5, thumb_y), (_LOG_SB_X + 5, thumb_y + 14), (80, 80, 180), -1)
            cv2.rectangle(canvas, (_LOG_SB_X - 5, thumb_y), (_LOG_SB_X + 5, thumb_y + 14), (120, 120, 220), 1)

            # Summary Modal
            if summary_card_visible:
                card_x1, card_y1 = CAM_X + 310, CAM_Y + 230
                card_x2, card_y2 = CAM_X + 620, CAM_Y + 460
                cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (40, 40, 40), -1)
                cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), COLOR_GREEN, 2)
                cv2.rectangle(canvas, (card_x2 - 125, card_y2 - 28), (card_x2 - 15, card_y2 - 8), (0, 150, 70), -1)
                cv2.rectangle(canvas, (card_x2 - 125, card_y2 - 28), (card_x2 - 15, card_y2 - 8), COLOR_REAL_WHITE, 1)

            # --- PIL text rendering ---
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(canvas_rgb)
            draw = ImageDraw.Draw(pil_img)

            rgb_title = (COLOR_TITLE[2], COLOR_TITLE[1], COLOR_TITLE[0])
            rgb_text  = (COLOR_TEXT_MAIN[2], COLOR_TEXT_MAIN[1], COLOR_TEXT_MAIN[0])
            rgb_grey  = (COLOR_GREY[2], COLOR_GREY[1], COLOR_GREY[0])
            rgb_green = (COLOR_GREEN[2], COLOR_GREEN[1], COLOR_GREEN[0])
            rgb_red   = (COLOR_RED[2], COLOR_RED[1], COLOR_RED[0])
            rgb_white = (255, 255, 255)

            draw.text((20, 11), "INDUSTRIAL SOP MONITORING SYSTEM", font=font_title, fill=rgb_title)
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw.text((CANVAS_W - 190, 13), timestamp_str, font=font_body, fill=rgb_grey)

            draw.text((CAM_X + 20, prog_y + 8), f"SOP Workflow Progress: {tot_prog}%", font=font_header, fill=rgb_text)
            draw.text((CAM_X, prog_y + 40), "SOP REFERENCE FRAMES", font=font_header, fill=rgb_title)
            for display_idx in range(4):
                i = scroll_pos + display_idx
                if i >= 8: continue
                rx = CAM_X + display_idx * 160
                draw.text((rx + 8, ref_y + 72), step_names[i], font=font_body, fill=rgb_text)

            # Stats Cards
            headers   = ["Detection Count", "OK Count", "NG Count", "Yield"]
            yield_val = (ok_count / max(1, det_count)) * 100
            vals      = [str(det_count), str(ok_count), str(ng_count), f"{yield_val:.1f}%"]
            cols_rgb  = [rgb_text, rgb_green, rgb_red, rgb_green]
            for i in range(4):
                cx = rx_start + i * 145
                draw.text((cx + 10, CAM_Y + 15), headers[i], font=font_stats_lbl, fill=rgb_grey)
                draw.text((cx + 15, CAM_Y + 45), vals[i],    font=font_stats_val, fill=cols_rgb[i])

            draw.text((rx_start + 15, vy + 12), "VIDEO SOURCE & PERFORMANCE", font=font_header, fill=rgb_title)
            draw.text((rx_start + 15, vy + 42), "Source: Live Camera (0)",    font=font_body,   fill=rgb_text)
            draw.text((rx_start + 15, vy + 68), f"FPS: {1/max(delta, 1e-6):02.1f}", font=font_body, fill=rgb_text)
            draw.text((rx_start + 300, vy + 42), "GPU Load: 12% (Active)",    font=font_body,   fill=rgb_text)
            draw.text((rx_start + 300, vy + 68), "Memory: 1.4 GB",            font=font_body,   fill=rgb_text)

            draw.text((rx_start + 15, my + 12), "MES SYSTEM CONNECTION",      font=font_header, fill=rgb_title)
            draw.text((rx_start + 25, my + 46), "CONNECTED",                  font=font_body,   fill=rgb_white)
            draw.text((rx_start + 140, my + 44), "Server: http://mes-server/api/inspect-op", font=font_body, fill=rgb_text)
            draw.text((rx_start + 15, my + 68), "Protocol: MQTT / TCP-IP",    font=font_body,   fill=rgb_text)

            draw.text((rx_start + 15, ey + 12), "EMPLOYEE WORK STATION",      font=font_header, fill=rgb_title)
            draw.text((rx_start + 15, ey + 42), "Worker ID: EMP001",          font=font_body,   fill=rgb_text)
            draw.text((rx_start + 15, ey + 68), "Station: STATION-3",         font=font_body,   fill=rgb_text)
            draw.text((rx_start + 300, ey + 42), "Shift: A (Day Shift)",      font=font_body,   fill=rgb_text)
            draw.text((rx_start + 300, ey + 68), "Validation: PASS",          font=font_body,   fill=rgb_green)

            draw.text((rx_start + 15, ly + 12), "REAL-TIME WORKFLOW EVENT LOG", font=font_header, fill=rgb_title)

            # Render only the visible log lines, clipped inside the box
            visible_logs = log_book.logs[log_scroll_pos: log_scroll_pos + LOG_VISIBLE]
            for line_idx, msg in enumerate(visible_logs):
                draw.text((rx_start + 15, ly + 32 + line_idx * LOG_LINE_H), msg, font=font_log, fill=rgb_text)

            if summary_card_visible:
                draw.text((card_x1 + 15, card_y1 + 15), "ACTION COMPLETED", font=font_header, fill=rgb_green)
                stats = [
                    ("Worker ID",      "EMP001"),
                    ("Current Action", summary_action),
                    ("Hand",           summary_hand),
                    ("Direction",      summary_direction),
                    ("Rotation Count", f"{summary_turns:.2f} Turns" if active_step == 5 else "N/A"),
                    ("Completion",     "100%"),
                    ("Status",         "Completed"),
                ]
                y_offset = card_y1 + 48
                for key, val in stats:
                    draw.text((card_x1 + 15, y_offset), f"{key:<15}: {val}", font=font_body, fill=rgb_white)
                    y_offset += 15
                draw.text((card_x2 - 115, card_y2 - 24), "OK (Press Enter)", font=font_body, fill=rgb_white)

            # Convert PIL image back to BGR for OpenCV display
            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            cv2.imshow("Industrial Monitor", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 13:  # Enter key
                if not summary_card_visible:
                    summary_card_visible = True
                    step_progress[active_step - 1] = 100
                    summary_action = f"Manual OK: {step_names[active_step - 1]}"
                    summary_hand = "N/A"
                    summary_direction = "N/A"
                    summary_turns = 0.0
                else:
                    # Proceed to next SOP step
                    summary_card_visible = False
                    tracker.reset()
                    
                    det_count += 1
                    ok_count += 1
                    
                    if active_step < 8:
                        active_step += 1
                        log_book.add(f"System switched to Step {active_step}: {step_names[active_step - 1]}.")
                    else:
                        active_step = 1
                        step_progress = [0] * 8
                        weight_timer = 0.0
                        point_timer = 0.0
                        log_book.add("SOP workflow completed. Resetting to Step 1.")
                        
                    current_scroll = global_scroll_pos
                    target_scroll = max(0, min(active_step - 1, 4))
                    if active_step - 1 >= current_scroll + 4:
                        target_scroll = (active_step - 1) - 3
                    elif active_step - 1 < current_scroll:
                        target_scroll = active_step - 1
                    else:
                        target_scroll = current_scroll
                    global_scroll_pos = target_scroll
                        
            elif key == ord("n"):  # Simulate NG (fail step)
                if not summary_card_visible:
                    det_count += 1
                    ng_count += 1
                    log_book.add("Inspection mismatch detected: Step marked NG.")

        cap.release()
        cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification...")
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")
    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    model = build_model(conf)
    transform = Demo.get_transform_for_inf(conf.test_transforms)
    
    # Try loading checkpoint if config specifies it
    if conf.model.checkpoint is not None and os.path.exists(conf.model.checkpoint):
        try:
            snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
            model.load_state_dict(snapshot["MODEL_STATE"])
            logging.info("Trained model checkpoint loaded successfully.")
        except Exception as e:
            logging.warning(f"Could not load checkpoint: {e}. Falling back to untrained baseline.")
    
    if model is not None:
        model.eval()
        Demo.run(model, transform, conf)
