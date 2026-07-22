"""Geometric rule-based gesture classifier using MediaPipe landmarks.

Implements the scale-invariant finger-extension heuristics described in the
project specification. The wrist->middle-MCP distance is used as the scale
factor so the thresholds are independent of hand size or distance to the
camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Landmark indices defined by MediaPipe Hands.
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17

# Relative thresholds (fraction of wrist->middle-MCP scale).
FINGER_EXTEND_THRESHOLD = 0.52
THUMB_TIP_MCP_THRESHOLD = 0.38
THUMB_TIP_MIDDLE_MCP_THRESHOLD = 0.65


@dataclass
class LandmarkFrame:
    """Container for a single hand's 21 landmarks."""

    landmarks: np.ndarray  # shape (21, 3)
    handedness: str = "Right"  # "Left" or "Right"

    @classmethod
    def from_list(cls, points: Sequence[Sequence[float]], handedness: str = "Right") -> "LandmarkFrame":
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (21, 3):
            raise ValueError(f"Expected 21x3 landmarks, got {arr.shape}")
        return cls(landmarks=arr, handedness=handedness)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _scale(lm: np.ndarray) -> float:
    """Wrist -> middle MCP distance, used as the size-invariant scale."""
    return max(_distance(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)


def _finger_extended(lm: np.ndarray, tip: int, mcp: int, scale: float) -> bool:
    return _distance(lm[tip], lm[mcp]) > FINGER_EXTEND_THRESHOLD * scale


def _thumb_extended(lm: np.ndarray, scale: float) -> bool:
    dist_tip_index_mcp = _distance(lm[THUMB_TIP], lm[INDEX_MCP])
    dist_tip_middle_mcp = _distance(lm[THUMB_TIP], lm[MIDDLE_MCP])
    return (
        dist_tip_index_mcp > THUMB_TIP_MCP_THRESHOLD * scale
        and dist_tip_middle_mcp > THUMB_TIP_MIDDLE_MCP_THRESHOLD * scale
    )


@dataclass
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool


def finger_states(lm: np.ndarray) -> FingerState:
    scale = _scale(lm)
    return FingerState(
        thumb=_thumb_extended(lm, scale),
        index=_finger_extended(lm, INDEX_TIP, INDEX_MCP, scale),
        middle=_finger_extended(lm, MIDDLE_TIP, MIDDLE_MCP, scale),
        ring=_finger_extended(lm, RING_TIP, RING_MCP, scale),
        pinky=_finger_extended(lm, PINKY_TIP, PINKY_MCP, scale),
    )


def _thumb_pointing_up(lm: np.ndarray) -> bool:
    """Thumb tip is above wrist (smaller y = higher in image coordinates)."""
    return lm[THUMB_TIP][1] < lm[WRIST][1]


def _thumb_pointing_down(lm: np.ndarray) -> bool:
    """Thumb tip is below wrist (larger y = lower in image coordinates)."""
    return lm[THUMB_TIP][1] > lm[WRIST][1]


def _ok_sign(lm: np.ndarray, state: FingerState) -> bool:
    """Thumb tip near index tip, middle/ring/pinky extended."""
    contact = _distance(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.35 * _scale(lm)
    return contact and state.middle and state.ring and state.pinky


def _call_sign(lm: np.ndarray, state: FingerState) -> bool:
    """Thumb and pinky extended, index/middle/ring folded."""
    return state.thumb and state.pinky and not (state.index or state.middle or state.ring)


def _rock_sign(lm: np.ndarray, state: FingerState) -> bool:
    """Index and pinky extended, middle/ring folded, thumb folded."""
    return state.index and state.pinky and not (state.middle or state.ring or state.thumb)


def classify_gesture(landmarks: np.ndarray) -> str:
    """Classify a 21x3 landmark array into one of the gesture names.

    Returns the gesture name matching the gestures shown in gestures.png:
      ok, call, rock, like (thumbs_up), dislike (thumbs_down),
      palm, four, three, peace, one, fist, thumb_index,
      little_finger, middle_finger, no_gesture.
    """
    if landmarks is None or landmarks.shape != (21, 3):
        return "no_gesture"

    state = finger_states(landmarks)
    i, m, r, p, t = state.index, state.middle, state.ring, state.pinky, state.thumb

    # --- Priority: contact-based gestures first ---

    # OK: thumb tip near index tip, rest extended
    if _ok_sign(landmarks, state):
        return "ok"

    # Call: thumb + pinky only
    if _call_sign(landmarks, state):
        return "call"

    # Rock / horns: index + pinky only (no thumb)
    if _rock_sign(landmarks, state):
        return "rock"

    # --- Thumb-only gestures (like / dislike) ---
    if t and not (i or m or r or p):
        if _thumb_pointing_up(landmarks):
            return "like"    # thumbs up
        else:
            return "dislike" # thumbs down

    # --- All-fingers-extended gestures (palm / four / stop) ---
    # Palm: all 4 fingers extended (thumb may or may not be extended)
    # This correctly catches the open-hand / palm gesture for both
    # "stop" (palm toward camera) and open hand.
    if i and m and r and p:
        # All four main fingers extended = palm (stop gesture).
        # With or without thumb extended both count as palm.
        return "palm"

    # Four fingers without thumb = four
    # (This is now handled above since we return palm for i+m+r+p regardless)

    # --- Three finger combinations ---
    # Peace / two_up: index + middle up (with or without thumb)
    if i and m and not r and not p:
        return "peace"

    # Three: index + middle + ring (no pinky)
    if i and m and r and not p:
        return "three"

    # Three with thumb: treated as three
    if i and m and r and t and not p:
        return "three"

    # --- Single / two finger gestures ---
    # One: only index finger
    if i and not (m or r or p or t):
        return "one"

    # Thumb + index: gun / thumb_index shape
    if t and i and not (m or r or p):
        return "thumb_index"

    # Little / pinky finger only
    if p and not (i or m or r or t):
        return "little_finger"

    # Middle finger only
    if m and not (i or r or p or t):
        return "middle_finger"

    # Fist: no fingers extended at all
    if not (i or m or r or p or t):
        return "fist"

    return "no_gesture"


def classify_landmark_frame(frame: LandmarkFrame) -> str:
    return classify_gesture(frame.landmarks)


def handedness_label(label: str) -> str:
    """Normalise MediaPipe handedness into Left/Right."""
    if not label:
        return "Right"
    label = label.capitalize()
    return label if label in ("Left", "Right") else "Right"
