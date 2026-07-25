import cv2
import time
from collections import deque

# ------------------ CONFIG ------------------
CAMERA_INDEX = 2         # 0 = Laptop Camera, 2 = USB Camera (change based on detect_cameras.py result)
SMOOTHING_WINDOW = 30     # number of frames used to average FPS (higher = smoother, less jumpy)
WINDOW_NAME = "Camera FPS Test"

# Common resolutions to test, HIGHEST first. Camera will use the best one it supports.
RESOLUTIONS_TO_TEST = [
    (3840, 2160),   # 4K
    (2560, 1440),   # QHD
    (1920, 1080),   # Full HD
    (1600, 900),
    (1280, 720),    # HD
    (1024, 768),
    (800, 600),
    (640, 480),     # fallback
]
# ---------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if cv2.os.name == "nt" else 0)

if not cap.isOpened():
    print(f"❌ Unable to open camera with index {CAMERA_INDEX}")
    exit()

# Force MJPG (compressed) format instead of raw YUYV.
# Most USB cameras can only hit high FPS at high resolution using MJPG,
# because raw formats need far more USB bandwidth.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


def detect_max_resolution(cap):
    """Try resolutions from highest to lowest, return the best one the camera actually accepts."""
    best_width, best_height = 0, 0

    for w, h in RESOLUTIONS_TO_TEST:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Confirm the camera can actually deliver a frame at this size
        ret, frame = cap.read()
        if ret and frame is not None:
            best_width, best_height = actual_w, actual_h
            break  # first one that works, since list is sorted highest -> lowest

    return best_width, best_height


print("=" * 50)
print("Detecting maximum supported resolution...")
max_width, max_height = detect_max_resolution(cap)
print(f"✅ Using resolution : {max_width} x {max_height}")
print("=" * 50)

camera_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera Index        : {CAMERA_INDEX}")
print(f"Resolution           : {max_width} x {max_height}")
print(f"Reported Camera FPS  : {camera_fps:.2f}")
print("=" * 50)

# Set up fullscreen window (matches system/monitor full size)
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Rolling window of frame timestamps -> used to compute a SMOOTH fps
frame_times = deque(maxlen=SMOOTHING_WINDOW)

# For final summary stats
all_fps_samples = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame.")
        break

    current_time = time.time()
    frame_times.append(current_time)

    # Smooth FPS = (number of frames in window - 1) / (time span of window)
    if len(frame_times) >= 2:
        time_span = frame_times[-1] - frame_times[0]
        smooth_fps = (len(frame_times) - 1) / time_span if time_span > 0 else 0.0
    else:
        smooth_fps = 0.0

    if smooth_fps > 0:
        all_fps_samples.append(smooth_fps)

    frame_h, frame_w = frame.shape[:2]

    # Display FPS (smoothed, stable reading)
    cv2.putText(
        frame,
        f"Smoothed FPS : {smooth_fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Reported FPS : {camera_fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Resolution : {frame_w} x {frame_h}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 200, 255),
        2,
    )

    cv2.putText(
        frame,
        "Press 'q' to quit  |  Press 'f' to toggle fullscreen",
        (20, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # Toggle fullscreen on/off
        current_mode = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        new_mode = cv2.WINDOW_NORMAL if current_mode == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, new_mode)

cap.release()
cv2.destroyAllWindows()

print("\n========== RESULT ==========")
print(f"Resolution Used     : {max_width} x {max_height}")
print(f"Reported Camera FPS : {camera_fps:.2f}")

if all_fps_samples:
    avg_fps = sum(all_fps_samples) / len(all_fps_samples)
    min_fps = min(all_fps_samples)
    max_fps = max(all_fps_samples)
    print(f"Average Live FPS    : {avg_fps:.2f}")
    print(f"Min Live FPS        : {min_fps:.2f}")
    print(f"Max Live FPS        : {max_fps:.2f}")
else:
    print("No FPS samples were collected.")
print("============================")