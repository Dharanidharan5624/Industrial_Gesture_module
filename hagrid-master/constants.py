"""Class list and index definitions shared across the HAGRID system.

The HAGRIDv2 gesture taxonomy defines 18 gesture classes plus the special
``no_gesture`` category. Index values mirror the ordering used by the
released checkpoints so weights can be loaded without remapping.
"""

# ── Legacy constants kept for backward-compatibility ───────────────────────
# demo_ff.py and other original scripts import these directly.
IMAGES = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

targets = {
    0: "grabbing",
    1: "grip",
    2: "holy",
    3: "point",
    4: "call",
    5: "three3",
    6: "timeout",
    7: "xsign",
    8: "hand_heart",
    9: "hand_heart2",
    10: "little_finger",
    11: "middle_finger",
    12: "take_picture",
    13: "dislike",
    14: "fist",
    15: "four",
    16: "like",
    17: "mute",
    18: "ok",
    19: "one",
    20: "palm",
    21: "peace",
    22: "peace_inverted",
    23: "rock",
    24: "stop",
    25: "stop_inverted",
    26: "three",
    27: "three2",
    28: "two_up",
    29: "two_up_inverted",
    30: "three_gun",
    31: "thumb_index",
    32: "thumb_index2",
    33: "no_gesture",
}
# ── End legacy constants ───────────────────────────────────────────────────

# Primary gesture class names. The trailing entry is the "no gesture"
# background class which the geometric classifier also emits.
GESTURE_CLASSES = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "ok",
    "thumbs_up",
    "thumbs_down",
    "fist",
    "palm",
    "peace",
    "rock",
    "call",
    "middle_finger",
    "little_finger",
    "thumb_index",
    "no_gesture",
]

# Human readable aliases used by the geometric classifier and the UI.
GEOMETRIC_GESTURES = [
    "ok",
    "like",
    "dislike",
    "fist",
    "palm",
    "one",
    "peace",
    "three",
    "four",
    "rock",
    "call",
    "middle_finger",
    "little_finger",
    "thumb_index",
    "no_gesture",
]

# Convenience mapping from friendly name -> canonical checkpoint class name.
GESTURE_NAME_TO_INDEX = {name: idx for idx, name in enumerate(GESTURE_CLASSES)}
GESTURE_INDEX_TO_NAME = {idx: name for idx, name in enumerate(GESTURE_CLASSES)}

# Number of input channels for the backbone models.
IN_CHANNELS = 3
# Default input size (height, width) used by the full-frame classifier.
DEFAULT_FRAME_SIZE = (224, 224)
# Number of landmark coordinates produced by MediaPipe (21 points * 3 xyz).
NUM_LANDMARKS = 21
LANDMARK_DIM = 3

# Industrial monitoring constants --------------------------------------------
SCREW_CENTER = (320, 240)        # virtual screw target (frame centre)
ALIGN_THRESHOLD_PX = 50          # distance for ALIGNED state
GLOVE_SKIN_RATIO_THRESHOLD = 0.2  # <20% skin -> Glove detected
ROTATION_TURN_TARGET = 2.5       # turns required to flag "tight"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEFAULT_WORKER_ID = "EMP001"
DEFAULT_SOURCE = 0

# Logging
LOG_CSV_PATH = "screw_monitoring_log.csv"
LOG_DIR = "logs"
SCREENSHOT_PREFIX = "screenshot"
CSV_COLUMNS = [
    "Timestamp",
    "Worker_ID",
    "Action",
    "Object",
    "Tool",
    "Hand",
    "Direction",
    "Rotation_Count",
    "Confidence",
    "Status",
    "Screenshot_Path",
]

# Tool recognition (SIFT + FLANN)
FLANN_RATIO_THRESHOLD = 0.7
TOOL_MATCH_MIN_INLIERS = 10

# UI colour palette (BGR tuples for OpenCV overlays).
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 220)
COLOR_BLUE = (220, 130, 0)
COLOR_YELLOW = (0, 220, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GLOVE = (200, 180, 0)
COLOR_NORMAL = (120, 120, 120)
