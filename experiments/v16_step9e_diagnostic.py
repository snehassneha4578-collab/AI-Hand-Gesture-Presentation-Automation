import cv2
import time
import csv
import numpy as np
import mediapipe as mp
from collections import deque

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL = BASE + r"\models\hand_landmarker.task"
CSV_PATH = BASE + r"\dataset\v16_calibration\v16_personal_calibration.csv"

print("=" * 70)
print("V16 STEP 9E - LIVE GESTURE SCORE DIAGNOSTIC")
print("=" * 70)

# ---------- Load calibration ----------
with open(CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]
data = rows[1:]

label_col = header.index("label")

X = np.array([
    [float(v) for i, v in enumerate(row)
     if i != label_col and header[i] != "session_id"]
    for row in data
], dtype=np.float32)

labels = np.array([row[label_col] for row in data])

# ---------- Same feature extraction ----------
def features(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks],
                   dtype=np.float32)

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
            out.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]))

    for i in tips:
        out.extend(pts[i].tolist())

    for i in [8, 12, 16, 20]:
        out.append(np.linalg.norm(pts[4] - pts[i]))

    return np.array(out, dtype=np.float32)

# ---------- Build prototypes ----------
prototypes = {}

for gesture in ["one", "two", "three"]:
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

        samples.append(features(landmarks))

    arr = np.vstack(samples)

    prototypes[gesture] = {
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0) + 0.05
    }

    print(f"Prototype loaded: {gesture} ({len(samples)} samples)")

# ---------- MediaPipe ----------
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam failed.")
    input("Press Enter...")
    raise SystemExit

print()
print("LIVE TEST")
print("Show ONE, then TWO, then THREE.")
print("Watch the terminal scores.")
print("Press Q to quit.")
print()

history = deque(maxlen=5)
frame_count = 0
last_print = 0

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(image)

        gesture = "NO HAND"

        if result.hand_landmarks:

            current = features(result.hand_landmarks[0])

            scores = {}

            for name, proto in prototypes.items():
                z = (current - proto["mean"]) / proto["std"]
                scores[name] = np.mean(np.abs(z))

            ranked = sorted(scores.items(), key=lambda x: x[1])

            gesture = ranked[0][0]

            margin = ranked[1][1] - ranked[0][1]

            history.append(gesture)

            stable = sum(x == gesture for x in history)

            now = time.time()

            if now - last_print >= 0.5:
                print(
                    f"Frame {frame_count:4d} | "
                    f"ONE={scores['one']:.2f} | "
                    f"TWO={scores['two']:.2f} | "
                    f"THREE={scores['three']:.2f} | "
                    f"Best={gesture.upper():5s} | "
                    f"Margin={margin:.2f} | "
                    f"Stable={stable}"
                )
                last_print = now

            cv2.putText(
                frame,
                f"Gesture: {gesture.upper()}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"One {scores['one']:.2f}  Two {scores['two']:.2f}  Three {scores['three']:.2f}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        else:
            history.clear()

            cv2.putText(
                frame,
                "NO HAND",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        cv2.imshow("V16 Step 9E - Gesture Diagnostic", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 70)
print("V16 STEP 9E COMPLETE")
print(f"Frames processed: {frame_count}")
print("=" * 70)

input("Press Enter to return to PowerShell...")
