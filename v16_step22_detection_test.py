import os
import cv2
import time
import numpy as np
import mediapipe as mp

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
TASK = os.path.join(BASE, "models", "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    landmarker.close()
    raise SystemExit

# Keep camera resolution consistent
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("="*70)
print("STEP 22 — IMPROVED REAL-TIME DETECTION TEST")
print("="*70)
print("")
print("Hold ONE, TWO and THREE for several seconds each.")
print("Move your hand slowly around the usable camera area.")
print("")
print("Press Q to finish.")
print("="*70)

frames = 0
hand_frames = 0
landmark_counts = []
start = time.time()

timestamp_ms = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    frames += 1

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Monotonically increasing timestamp for VIDEO mode
    timestamp_ms += int(1000 / 30)

    result = landmarker.detect_for_video(
        image,
        timestamp_ms
    )

    detected = len(result.hand_landmarks) > 0

    if detected:

        hand_frames += 1

        landmarks = result.hand_landmarks[0]

        landmark_counts.append(len(landmarks))

        h, w = frame.shape[:2]

        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]

        x1 = max(0, int(min(xs) * w))
        y1 = max(0, int(min(ys) * h))
        x2 = min(w-1, int(max(xs) * w))
        y2 = min(h-1, int(max(ys) * h))

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        # Draw all 21 landmarks
        for i,p in enumerate(landmarks):

            px = int(p.x * w)
            py = int(p.y * h)

            if 0 <= px < w and 0 <= py < h:

                cv2.circle(
                    frame,
                    (px,py),
                    4,
                    (0,255,255),
                    -1
                )

                cv2.putText(
                    frame,
                    str(i),
                    (px+5,py-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255,255,255),
                    1
                )

        status = "HAND: DETECTED"

    else:

        status = "HAND: NOT DETECTED"

    elapsed = time.time() - start

    fps = frames / max(elapsed, 1e-6)

    visibility = (
        hand_frames / frames * 100
        if frames else 0
    )

    cv2.putText(
        frame,
        status,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0) if detected else (0,0,255),
        2
    )

    cv2.putText(
        frame,
        f"Landmarks: {len(result.hand_landmarks[0]) if detected else 0}/21",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Visibility: {visibility:.1f}%",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Q = quit",
        (20,180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow(
        "STEP 22 - Improved MediaPipe Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

elapsed = time.time() - start

visibility = (
    hand_frames / frames * 100
    if frames else 0
)

fps = frames / max(elapsed,1e-6)

print("")
print("="*70)
print("STEP 22 RESULTS")
print("="*70)
print(f"Frames processed : {frames}")
print(f"Hand frames      : {hand_frames}")
print(f"Hand visibility  : {visibility:.2f}%")
print(f"Average FPS      : {fps:.2f}")

if landmark_counts:
    print(
        f"Landmarks detected: "
        f"{min(landmark_counts)}–{max(landmark_counts)}"
    )

print("")
print("COMPARISON")
print("Step 21 visibility : 71.81%")
print(f"Step 22 visibility : {visibility:.2f}%")
print(f"Improvement        : {visibility - 71.81:+.2f} percentage points")

if visibility >= 85:
    print("")
    print("EXCELLENT: Detection layer is now strong enough.")
    print("Next: rerun controlled gesture accuracy.")
elif visibility >= 75:
    print("")
    print("IMPROVED: Detection is better, but still not ideal.")
    print("Next: optimize recognition together with detection.")
else:
    print("")
    print("STILL WEAK: Detection remains a significant bottleneck.")
    print("Next: investigate camera framing/lighting and hand visibility.")

print("="*70)
