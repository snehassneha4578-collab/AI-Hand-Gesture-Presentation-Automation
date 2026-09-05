import cv2
import csv
import time
import numpy as np
import mediapipe as mp
from collections import deque

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL = BASE + r"\models\hand_landmarker.task"
CSV_PATH = BASE + r"\dataset\v16_calibration\v16_personal_calibration.csv"

print("=" * 70)
print("V16 STEP 9F - CONSERVATIVE LIVE GESTURE ACCEPTANCE")
print("=" * 70)

# ------------------------------------------------------------
# Load calibration
# ------------------------------------------------------------
with open(CSV_PATH, newline="") as f:
    rows = list(csv.reader(f))

header = rows[0]
data = rows[1:]

label_col = header.index("label")

X = np.array([
    [
        float(v)
        for i, v in enumerate(row)
        if i != label_col and header[i] != "session_id"
    ]
    for row in data
], dtype=np.float32)

labels = np.array([row[label_col] for row in data])

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def make_features(landmarks):

    pts = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32
    )

    pts = pts - pts[0]

    scale = np.linalg.norm(pts[9])

    if scale < 1e-6:
        scale = 1.0

    pts = pts / scale

    out = list(pts.flatten())

    tips = [4, 8, 12, 16, 20]

    for i in tips:
        out.append(np.linalg.norm(pts[i]))

    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            out.append(
                np.linalg.norm(pts[tips[i]] - pts[tips[j]])
            )

    for i in tips:
        out.extend(pts[i].tolist())

    for i in [8, 12, 16, 20]:
        out.append(np.linalg.norm(pts[4] - pts[i]))

    return np.array(out, dtype=np.float32)


# ------------------------------------------------------------
# Build personalized prototypes
# ------------------------------------------------------------
gestures = ["one", "two", "three"]

prototypes = {}

for gesture in gestures:

    samples = []

    for row, label in zip(X, labels):

        if label != gesture:
            continue

        landmarks = []

        for i in range(21):

            class P:
                pass

            p = P()
            p.x = row[i * 3]
            p.y = row[i * 3 + 1]
            p.z = row[i * 3 + 2]

            landmarks.append(p)

        samples.append(make_features(landmarks))

    arr = np.vstack(samples)

    prototypes[gesture] = {
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0) + 0.05
    }

    print(f"Loaded {gesture}: {len(samples)} samples")


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
MARGIN_THRESHOLD = 0.25
STABLE_REQUIRED = 5
COOLDOWN_SECONDS = 1.5

history = deque(maxlen=7)

last_action = "NONE"
last_action_time = 0

# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL
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
print()
print("Gesture mapping TEST:")
print("ONE   -> candidate command")
print("TWO   -> candidate command")
print("THREE -> candidate command")
print()
print(f"Margin threshold : {MARGIN_THRESHOLD}")
print(f"Stable frames    : {STABLE_REQUIRED}")
print(f"Cooldown         : {COOLDOWN_SECONDS} seconds")
print()
print("IMPORTANT: No PowerPoint actions are connected yet.")
print("Press Q to quit.")
print()

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(image)

        gesture = "NO HAND"
        margin = 0.0
        distance = 0.0
        stable = 0
        accepted = False

        if result.hand_landmarks:

            current = make_features(
                result.hand_landmarks[0]
            )

            scores = {}

            for name, proto in prototypes.items():

                z = (
                    current - proto["mean"]
                ) / proto["std"]

                scores[name] = np.mean(np.abs(z))

            ranked = sorted(
                scores.items(),
                key=lambda x: x[1]
            )

            gesture = ranked[0][0]
            distance = ranked[0][1]

            if len(ranked) >= 2:
                margin = ranked[1][1] - ranked[0][1]

            history.append(gesture)

            stable = sum(
                x == gesture
                for x in history
            )

            if (
                margin >= MARGIN_THRESHOLD
                and stable >= STABLE_REQUIRED
            ):

                now = time.time()

                if now - last_action_time >= COOLDOWN_SECONDS:

                    accepted = True
                    last_action = gesture.upper()
                    last_action_time = now

                    print(
                        f"ACCEPTED -> {last_action} | "
                        f"margin={margin:.2f} | "
                        f"distance={distance:.2f}"
                    )

        else:

            history.clear()

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------
        cv2.putText(
            frame,
            f"Gesture: {gesture.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
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
            f"Stable: {stable}/{STABLE_REQUIRED}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        status = "ACCEPTED" if accepted else "WAITING"

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if accepted else (0, 200, 255),
            2
        )

        cv2.putText(
            frame,
            f"Last: {last_action}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "V16 Step 9F - Acceptance Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 70)
print("V16 STEP 9F COMPLETE")
print("=" * 70)

input("Press Enter to return to PowerShell...")
