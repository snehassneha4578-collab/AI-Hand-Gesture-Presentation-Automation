import cv2
import csv
import time
import os
import numpy as np
import mediapipe as mp
from datetime import datetime

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL_PATH = BASE + r"\models\hand_landmarker.task"
OUT_DIR = BASE + r"\dataset\v16_calibration"
OUT_FILE = OUT_DIR + r"\v16_live_calibration_one_two_three.csv"

os.makedirs(OUT_DIR, exist_ok=True)

GESTURES = ["one", "two", "three"]
SAMPLES_PER_GESTURE = 100

print("=" * 70)
print("V16 STEP 9G - LIVE PERSONAL CALIBRATION")
print("=" * 70)

print()
print("This calibration uses your ACTUAL webcam environment.")
print()
print("You will collect:")
print("ONE   = 100 samples")
print("TWO   = 100 samples")
print("THREE = 100 samples")
print()
print("Keep your hand comfortably in front of the camera.")
print("Move naturally while maintaining the gesture.")
print()
print("Press Q to cancel.")
print()

# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    input("Press Enter...")
    raise SystemExit

all_rows = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for gesture in GESTURES:

        print()
        print("=" * 70)
        print(f"GET READY: {gesture.upper()}")
        print("=" * 70)
        print("Hold the gesture in front of the camera.")
        print("Collection begins automatically.")
        print()

        start_wait = time.time()

        while time.time() - start_wait < 3:
            ret, frame = cap.read()

            if not ret:
                continue

            remaining = 3 - (time.time() - start_wait)

            cv2.putText(
                frame,
                f"GET READY: {gesture.upper()}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Starting in {remaining:.1f}",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("V16 Live Calibration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        collected = 0

        while collected < SAMPLES_PER_GESTURE:

            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(image)

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                row = []

                for p in landmarks:
                    row.extend([p.x, p.y, p.z])

                all_rows.append(
                    [datetime.now().isoformat(), gesture] + row
                )

                collected += 1

            cv2.putText(
                frame,
                f"Gesture: {gesture.upper()}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Keep gesture steady - Q = Cancel",
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow("V16 Live Calibration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
header = ["timestamp", "label"]

for i in range(21):
    header += [f"x_{i}", f"y_{i}", f"z_{i}"]

with open(OUT_FILE, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print()
print("=" * 70)
print("V16 STEP 9G COMPLETE")
print("=" * 70)
print(f"Total samples: {len(all_rows)}")
print(f"Saved to:")
print(OUT_FILE)
print()
print("ONE   :", sum(r[1] == "one" for r in all_rows))
print("TWO   :", sum(r[1] == "two" for r in all_rows))
print("THREE :", sum(r[1] == "three" for r in all_rows))
print("=" * 70)

input("Press Enter to return to PowerShell...")
