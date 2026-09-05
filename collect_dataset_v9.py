import cv2
import mediapipe as mp
import csv
import os
import time
import uuid
import numpy as np
from datetime import datetime

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")

GESTURES = ["fist", "one", "two", "three", "open_palm", "pinch"]

TARGET_SAMPLES = 350
FRAME_INTERVAL = 6
MIN_MOVEMENT = 0.004
MAX_SESSION_SECONDS = 120

os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(hand):
    features = []

    for landmark in hand:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features

def get_last_session(label):
    if not os.path.exists(DATASET_PATH):
        return None

    try:
        with open(DATASET_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)

            sessions = []

            for row in reader:
                if row["label"] == label:
                    sessions.append(row["session_id"])

            if sessions:
                return sessions[-1]

    except Exception:
        pass

    return None

def collect_gesture(label, session_number):
    print("\n" + "=" * 75)
    print(f"V9 COLLECTION: {label.upper()} | SESSION {session_number}")
    print("=" * 75)

    print("""
IMPORTANT:
Keep the SAME gesture, but deliberately vary:

  POSITION:
    left / center / right
    upper / middle / lower

  DISTANCE:
    closer / normal / farther

  ORIENTATION:
    slight left tilt
    slight right tilt
    natural wrist rotation

  MOVEMENT:
    small natural movements
    slowly reposition the hand
    avoid holding one exact position

Do NOT change the gesture itself.
""")

    input("Press ENTER when ready...")

    session_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_S{session_number}_"
        + uuid.uuid4().hex[:6]
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    rows = []
    previous = None
    frame_count = 0
    start_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:

        while len(rows) < TARGET_SAMPLES:

            if time.time() - start_time > MAX_SESSION_SECONDS:
                print("\nMaximum session time reached.")
                break

            ret, frame = cap.read()

            if not ret:
                continue

            frame_count += 1

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                features = extract_features(hand)

                current = np.array(features)

                movement = 1.0

                if previous is not None:
                    movement = np.linalg.norm(current - previous)

                previous = current

                if (
                    frame_count % FRAME_INTERVAL == 0
                    and movement >= MIN_MOVEMENT
                ):
                    rows.append(
                        features + [
                            label,
                            session_id
                        ]
                    )

                for landmark in hand:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (0, 255, 0),
                        -1
                    )

            progress = len(rows) / TARGET_SAMPLES

            cv2.putText(
                frame,
                f"Gesture: {label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Session: {session_number}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {len(rows)}/{TARGET_SAMPLES}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Move hand: position + distance + rotation",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow("V9 Controlled Dataset Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nCollection stopped by user.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if not rows:
        print("No samples collected.")
        return

    file_exists = os.path.exists(DATASET_PATH)

    with open(DATASET_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            header = []

            for i in range(63):
                header.append(f"x_{i//3}" if i % 3 == 0 else
                              f"y_{i//3}" if i % 3 == 1 else
                              f"z_{i//3}")

            header += ["label", "session_id"]

            writer.writerow(header)

        writer.writerows(rows)

    print(
        f"\nSaved {len(rows)} samples for "
        f"{label} Session {session_number}"
    )

    print(f"Session ID: {session_id}")


print("\n" + "=" * 75)
print("V9 CONTROLLED MULTI-SESSION DATASET COLLECTION")
print("=" * 75)

print("""
This collector creates SESSION-INDEPENDENT training data.

For each gesture:
  Session 1 = one collection environment
  Session 2 = deliberately different conditions
  Session 3 = deliberately different conditions
  ...
""")

for session_number in range(1, 4):

    print("\n" + "#" * 75)
    print(f"STARTING SESSION {session_number}")
    print("#" * 75)

    for gesture in GESTURES:

        collect_gesture(
            gesture,
            session_number
        )

        print(
            "\nNext gesture will start after you press ENTER."
        )

        if gesture != GESTURES[-1]:
            input()

print("\n" + "=" * 75)
print("V9 COLLECTION COMPLETE")
print("=" * 75)

print(f"\nDataset saved at:\n{DATASET_PATH}")
