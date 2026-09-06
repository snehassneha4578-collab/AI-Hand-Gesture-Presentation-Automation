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
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    landmarker.close()
    raise SystemExit

print("="*70)
print("STEP 21 — LIVE LANDMARK / DETECTION DIAGNOSTIC")
print("="*70)
print("")
print("Hold ONE, TWO and THREE in front of the camera.")
print("Move your hand slowly closer/farther and left/right.")
print("")
print("Watch:")
print("  HAND = whether MediaPipe sees your hand")
print("  LANDMARKS = should always be 21 when detected")
print("  VISIBILITY = percentage of frames with a detected hand")
print("")
print("Press Q to finish.")
print("="*70)

frames = 0
hand_frames = 0
landmark_counts = []
fps_values = []

start = time.time()
last_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    frames += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(image)

    detected = len(result.hand_landmarks) > 0

    if detected:

        hand_frames += 1

        landmarks = result.hand_landmarks[0]

        landmark_counts.append(len(landmarks))

        # Bounding box
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]

        x1 = int(min(xs) * frame.shape[1])
        y1 = int(min(ys) * frame.shape[0])
        x2 = int(max(xs) * frame.shape[1])
        y2 = int(max(ys) * frame.shape[0])

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        # Draw all 21 landmarks
        for i,p in enumerate(landmarks):

            px = int(p.x * frame.shape[1])
            py = int(p.y * frame.shape[0])

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

        hand_text = "HAND: DETECTED"

        # Print wrist and fingertip coordinates occasionally
        if frames % 30 == 0:

            wrist = landmarks[0]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            print(
                f"Frame {frames}: "
                f"Wrist=({wrist.x:.3f},{wrist.y:.3f},{wrist.z:.3f}) | "
                f"IndexTip=({index_tip.x:.3f},{index_tip.y:.3f},{index_tip.z:.3f}) | "
                f"MiddleTip=({middle_tip.x:.3f},{middle_tip.y:.3f},{middle_tip.z:.3f}) | "
                f"RingTip=({ring_tip.x:.3f},{ring_tip.y:.3f},{ring_tip.z:.3f}) | "
                f"PinkyTip=({pinky_tip.x:.3f},{pinky_tip.y:.3f},{pinky_tip.z:.3f})"
            )

    else:

        hand_text = "HAND: NOT DETECTED"

    now = time.time()

    if now - last_time >= 1.0:

        current_fps = frames / max(now-start,1e-6)

        fps_values.append(current_fps)

        last_time = now

    visibility = (
        hand_frames / frames * 100
        if frames > 0 else 0
    )

    fps = (
        frames / max(time.time()-start,1e-6)
    )

    cv2.putText(
        frame,
        hand_text,
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
        f"Hand visibility: {visibility:.1f}%",
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
        "STEP 21 - MediaPipe Diagnostic",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

visibility = (
    hand_frames / frames * 100
    if frames else 0
)

avg_fps = (
    frames / max(time.time()-start,1e-6)
)

print("")
print("="*70)
print("STEP 21 DIAGNOSTIC RESULTS")
print("="*70)
print(f"Frames processed : {frames}")
print(f"Hand frames      : {hand_frames}")
print(f"Hand visibility  : {visibility:.2f}%")
print(f"Average FPS      : {avg_fps:.2f}")

if landmark_counts:
    print(
        f"Landmarks detected: "
        f"{min(landmark_counts)}–{max(landmark_counts)}"
    )
else:
    print("No landmarks detected.")

print("")
print("INTERPRETATION:")

if visibility >= 80:
    print("GOOD: MediaPipe detection is strong.")
    print("The main problem is likely feature/classification domain shift.")
elif visibility >= 60:
    print("MODERATE: MediaPipe detection is usable but inconsistent.")
    print("We should improve capture/detection conditions before ML tuning.")
else:
    print("WEAK: MediaPipe is missing the hand too often.")
    print("We should fix the live capture/detection pipeline first.")

print("="*70)
