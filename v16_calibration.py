import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
CALIBRATION_DIR = os.path.join(BASE_DIR, "dataset", "v16_calibration")

os.makedirs(CALIBRATION_DIR, exist_ok=True)

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]

SAMPLES_PER_GESTURE = 100


def extract_landmarks(hand_landmarks):

    features = []

    for landmark in hand_landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


print("=" * 70)
print("V16 PERSONALIZED CALIBRATION")
print("MediaPipe Tasks API")
print("=" * 70)

if not os.path.exists(MODEL_PATH):

    print("\nERROR: MediaPipe model not found:")
    print(MODEL_PATH)
    raise SystemExit


BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")
    landmarker.close()
    raise SystemExit


print("\nWebcam opened successfully.")

all_rows = []


for gesture in GESTURES:

    print("\n" + "=" * 70)
    print(f"NEXT GESTURE: {gesture.upper()}")
    print("=" * 70)

    input(
        f"\nPress ENTER when ready to collect '{gesture}'..."
    )

    print(
        f"\nCollecting {SAMPLES_PER_GESTURE} samples..."
    )

    collected = 0

    while collected < SAMPLES_PER_GESTURE:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        detected = False

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            features = extract_landmarks(hand)

            if len(features) == 63:

                row = features + [
                    gesture,
                    collected
                ]

                all_rows.append(row)

                collected += 1

                detected = True

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

        status = "HAND DETECTED" if detected else "SHOW HAND"

        cv2.putText(
            frame,
            status,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if detected else (0, 0, 255),
            2
        )

        cv2.imshow(
            "V16 Personalized Calibration",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()

            print("\nCalibration stopped by user.")
            raise SystemExit

    print(
        f"Completed {gesture}: "
        f"{collected}/{SAMPLES_PER_GESTURE}"
    )


cap.release()
cv2.destroyAllWindows()
landmarker.close()


columns = []

for i in range(21):

    columns.append(f"x_{i}")
    columns.append(f"y_{i}")
    columns.append(f"z_{i}")

columns.extend([
    "label",
    "sample_id"
])


df = pd.DataFrame(
    all_rows,
    columns=columns
)


output_file = os.path.join(
    CALIBRATION_DIR,
    "v16_personal_calibration.csv"
)


df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("V16 CALIBRATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal samples: {len(df)}"
)

print("\nSamples per gesture:")

print(
    df["label"].value_counts().to_string()
)

print(
    f"\nSaved to:\n{output_file}"
)
