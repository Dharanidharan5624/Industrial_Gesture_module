import cv2
import time

# Camera Index (0 = Laptop Camera, 1 = USB Camera, 2 = Another USB Camera...)
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"❌ Unable to open camera with index {CAMERA_INDEX}")
    exit()

# Camera reported FPS
camera_fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 50)
print(f"Camera Index        : {CAMERA_INDEX}")
print(f"Reported Camera FPS : {camera_fps:.2f}")
print("=" * 50)

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame.")
        break

    # Calculate Actual FPS
    current_time = time.time()
    actual_fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS
    cv2.putText(
        frame,
        f"Actual FPS : {actual_fps:.2f}",
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

    cv2.imshow("Camera FPS Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========== RESULT ==========")
print(f"Reported Camera FPS : {camera_fps:.2f}")
print(f"Live FPS (Last Frame): {actual_fps:.2f}")
print("============================")