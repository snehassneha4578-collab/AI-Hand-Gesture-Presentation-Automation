import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
from collections import deque

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL_PATH = BASE + r"\models\hand_landmarker.task"
CALIBRATION = BASE + r"\dataset\v16_calibration\v16_personal_calibration.csv"

print("=" * 70)
print("V16 STEP 9C - REAL-TIME PERSONALIZED RECOGNIZER")
print("=" * 70)

# ------------------------------------------------------------
# Load calibration data
# ------------------------------------------------------------
data = np.genfromtxt(CALIBRATION, delimiter=",", dtype=str)

header = data[0]
rows = data[1:]

# Find label column
label_idx = list(header).index("label")

labels = rows[:, label_idx]

feature_columns = [
    i for i, name in enumerate(header)
    if name not in ("label", "session_id")
]

X = rows[:, feature_columns].astype(float)

# ------------------------------------------------------------
# MediaPipe landmark normalization
# ------------------------------------------------------------
def make_features(landmarks):
    pts = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32
    )

    wrist = pts[0].copy()
    pts = pts - wrist

    scale = np.linalg.norm(pts[9])

    if scale < 1e-6:
        scale = 1.0

    pts = pts / scale

    features = list(pts.flatten())

    fingertips = [4, 8, 12, 16, 20]

    # Wrist-to-fingertip distances
    for idx in fingertips:
        features.append(np.linalg.norm(pts[idx]))

    # Pairwise fingertip distances
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            a = pts[fingertips[i]]
            b = pts[fingertips[j]]
            features.append(np.linalg.norm(a - b))

    # Relative fingertip positions
    for idx in fingertips:
        features.extend(pts[idx].tolist())

    # Thumb-to-fingertip distances
    thumb = pts[4]

    for idx in [8, 12, 16, 20]:
        features.append(np.linalg.norm(thumb - pts[idx]))

    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------
# Build personalized prototypes from calibration data
# ------------------------------------------------------------
gesture_features = {
    "one": [],
    "two": [],
    "three": []
}

# Calibration CSV already contains landmark coordinates.
# Use first 63 numerical landmark values.
for label in ["one", "two", "three"]:
    mask = labels == label
    samples = X[mask]

    for sample in samples:
        landmarks = []

        for i in range(21):
            class P:
                pass

            p = P()
            p.x = sample[i * 3]
            p.y = sample[i * 3 + 1]
            p.z = sample[i * 3 + 2]
            landmarks.append(p)

        gesture_features[label].append(make_features(landmarks))

prototypes = {}

for gesture, values in gesture_features.items():
    arr = np.vstack(values)

    prototypes[gesture] = {
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0) + 0.05
    }

    print(f"Loaded {gesture}: {len(values)} samples")

# ------------------------------------------------------------
# Real-time recognizer
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

print()
print("Webcam started.")
print("Show ONE, TWO or THREE.")
print("Press Q to quit.")
print("Acceptance: margin >= 1.0 AND 3 stable frames")
print()

history = deque(maxlen=5)

last_time = time.time()
fps = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        gesture = "NO HAND"
        margin = 0.0
        distance = 0.0
        stable = 0
        accepted = False

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            current = make_features(landmarks)

            scores = {}

            for name, proto in prototypes.items():

                z = (
                    current - proto["mean"]
                ) / proto["std"]

                scores[name] = np.mean(np.abs(z))

            ordered = sorted(
                scores.items(),
                key=lambda x: x[1]
            )

            gesture = ordered[0][0]
            distance = ordered[0][1]

            if len(ordered) >= 2:
                margin = ordered[1][1] - ordered[0][1]

            history.append(gesture)

            stable = sum(
                1 for x in history
                if x == gesture
            )

            accepted = (
                margin >= 1.0
                and stable >= 3
            )

        else:
            history.clear()

        # FPS
        now = time.time()
        dt = now - last_time

        if dt > 0:
            fps = 1.0 / dt

        last_time = now

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------
        cv2.putText(
            frame,
            f"Gesture: {gesture.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Margin: {margin:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Stable: {stable}/3",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        status = "ACCEPTED" if accepted else "WAITING"

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if accepted else (0, 200, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "V16 Step 9C - Personalized Recognizer",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 70)
print("V16 STEP 9C COMPLETE")
print("=" * 70)

input("Press Enter to return to PowerShell...")
