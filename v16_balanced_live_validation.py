import cv2
import mediapipe as mp
import csv
import time
from pathlib import Path
from datetime import datetime

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
OUT = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

MODEL = BASE / "models" / "hand_landmarker.task"

SAMPLES_PER_GESTURE = 100
GESTURES = [
    ("ONE", "O"),
    ("TWO", "T"),
    ("THREE", "H")
]

mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    input("Press Enter...")
    raise SystemExit

header = ["timestamp", "label"]

for i in range(21):
    header += [f"x_{i}", f"y_{i}", f"z_{i}"]

rows = []

print("="*72)
print("V16 BALANCED LIVE VALIDATION DATA COLLECTION")
print("="*72)
print()
print("We will collect:")
print("ONE   = 100 samples")
print("TWO   = 100 samples")
print("THREE = 100 samples")
print()
print("IMPORTANT:")
print("- Same webcam")
print("- Same distance")
print("- Same lighting")
print("- Keep the gesture clearly visible")
print("- This dataset will NOT be used for training")
print()
print("Controls:")
print("O = collect ONE")
print("T = collect TWO")
print("H = collect THREE")
print("Q = quit")
print()

for gesture, key in GESTURES:

    print("="*72)
    print(f"GET READY: {gesture}")
    print("="*72)
    print(f"Press {key} to start collecting {gesture}.")
    print()

    started = False
    count = 0

    while count < SAMPLES_PER_GESTURE:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"{gesture}: {count}/{SAMPLES_PER_GESTURE}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Press {key} to start | Q to quit",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.imshow("V16 Balanced Live Validation", frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            print("Collection cancelled.")
            input("Press Enter...")
            raise SystemExit

        if not started and k == ord(key.lower()):
            started = True
            print(f"Collecting {gesture}...")

        if not started:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            continue

        hand = result.hand_landmarks[0]

        values = []

        for landmark in hand:
            values.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        if len(values) != 63:
            continue

        rows.append([
            datetime.now().isoformat(),
            gesture.lower(),
            *values
        ])

        count += 1

        if count % 10 == 0:
            print(
                f"{gesture}: "
                f"{count}/{SAMPLES_PER_GESTURE}"
            )

        time.sleep(0.015)

    print(f"{gesture} COMPLETE: {count} samples")
    print()

cap.release()
cv2.destroyAllWindows()
landmarker.close()

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("="*72)
print("V16 BALANCED LIVE VALIDATION COMPLETE")
print("="*72)
print(f"Total samples: {len(rows)}")
print(f"Saved to:")
print(OUT)
print()
print("Expected:")
print("ONE   : 100")
print("TWO   : 100")
print("THREE : 100")
print()
print("DO NOT TRAIN ON THIS DATA.")
print("Next we will use it only for genuine validation.")
print("="*72)

input("Press Enter to return to PowerShell...")
