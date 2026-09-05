import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import time

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
TASK = BASE / "models" / "hand_landmarker.task"
OUT = BASE / "dataset" / "v16_calibration" / "v16_step9n_fresh_validation.csv"

GESTURES = ["one", "two", "three"]

FEATURE_NAMES = [
    "y_1",
    "pinky_ratio",
    "x_20",
    "x_12",
    "x_16",
    "y_2",
    "x_8",
    "x_1",
    "x_11",
    "middle_ratio",
    "y_17",
    "y_13",
    "y_18",
    "y_9",
    "middle_pinky",
    "x_17",
    "ring_pinky_x",
    "x_13",
    "x_15",
    "x_18"
]

def extract(pts):
    pts = pts.copy()
    pts -= pts[0]

    scale = np.linalg.norm(pts[9])

    if scale < 1e-6:
        scale = 1.0

    pts /= scale

    f = {}

    for i in range(21):
        f[f"x_{i}"] = pts[i, 0]
        f[f"y_{i}"] = pts[i, 1]
        f[f"z_{i}"] = pts[i, 2]

    fingertips = {
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20
    }

    mcps = {
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17
    }

    for name, idx in fingertips.items():
        f[f"{name}_ratio"] = (
            np.linalg.norm(pts[idx]) /
            max(np.linalg.norm(pts[mcps[name]]), 1e-6)
        )

    pairs = [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
        ("index", "ring"),
        ("index", "pinky"),
        ("middle", "pinky")
    ]

    for a, b in pairs:
        ia = fingertips[a]
        ib = fingertips[b]

        f[f"{a}_{b}"] = np.linalg.norm(
            pts[ia] - pts[ib]
        )

        f[f"{a}_{b}_x"] = pts[ia, 0] - pts[ib, 0]
        f[f"{a}_{b}_y"] = pts[ia, 1] - pts[ib, 1]

    for name, idx in fingertips.items():
        f[f"thumb_{name}"] = np.linalg.norm(
            pts[4] - pts[idx]
        )

    return f

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(TASK)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    input("Press Enter...")
    raise SystemExit

print("=" * 72)
print("V16 STEP 9N - FRESH GEOMETRY VALIDATION")
print("=" * 72)
print()
print("Each gesture will be recorded for 5 seconds.")
print()
print("O = ONE")
print("T = TWO")
print("H = THREE")
print("Q = quit")
print()
print("Keep your hand in the SAME position you normally use.")
print("=" * 72)

records = []

current = None
start_time = 0
duration = 5

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(image)

        if current is not None:

            elapsed = time.time() - start_time

            cv2.putText(
                frame,
                f"RECORDING: {current.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{max(0, duration-elapsed):.1f}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if result.hand_landmarks:

                pts = np.array(
                    [[p.x, p.y, p.z]
                     for p in result.hand_landmarks[0]],
                    dtype=np.float32
                )

                f = extract(pts)

                row = {
                    "label": current
                }

                for name in FEATURE_NAMES:
                    row[name] = f.get(name, np.nan)

                records.append(row)

            if elapsed >= duration:
                print(
                    f"{current.upper()} recording complete."
                )
                current = None

        else:

            cv2.putText(
                frame,
                "O=ONE   T=TWO   H=THREE   Q=QUIT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.imshow(
            "V16 Step 9N - Fresh Geometry Validation",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("o") and current is None:
            current = "one"
            start_time = time.time()
            print("ONE recording started...")

        elif key == ord("t") and current is None:
            current = "two"
            start_time = time.time()
            print("TWO recording started...")

        elif key == ord("h") and current is None:
            current = "three"
            start_time = time.time()
            print("THREE recording started...")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

fresh = pd.DataFrame(records)

if len(fresh) == 0:
    print("No samples collected.")
    input("Press Enter...")
    raise SystemExit

fresh.to_csv(OUT, index=False)

print()
print("=" * 72)
print("V16 STEP 9N RESULTS")
print("=" * 72)
print(f"Fresh samples: {len(fresh)}")
print(f"Saved to: {OUT}")
print()

for gesture in GESTURES:

    subset = fresh[fresh["label"] == gesture]

    print("-" * 72)
    print(f"{gesture.upper()} - {len(subset)} samples")
    print("-" * 72)

    for feature in FEATURE_NAMES:

        if feature not in subset:
            continue

        values = subset[feature].dropna()

        if len(values):
            print(
                f"{feature:25s} "
                f"mean={values.mean():8.4f} "
                f"std={values.std():8.4f}"
            )

print()
print("=" * 72)
print("PAIRWISE FRESH MEAN DIFFERENCES")
print("=" * 72)

for g1, g2 in [
    ("one", "two"),
    ("two", "three"),
    ("one", "three")
]:

    print()
    print(f"{g1.upper()} vs {g2.upper()}")

    scores = []

    for feature in FEATURE_NAMES:

        a = fresh.loc[
            fresh["label"] == g1, feature
        ].dropna()

        b = fresh.loc[
            fresh["label"] == g2, feature
        ].dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        pooled = np.sqrt(
            (a.var() + b.var()) / 2
        )

        if pooled < 1e-9:
            score = 0
        else:
            score = abs(
                a.mean() - b.mean()
            ) / pooled

        scores.append(
            (feature, score, a.mean(), b.mean())
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for feature, score, a, b in scores[:10]:
        print(
            f"{feature:25s} "
            f"{score:7.3f} | "
            f"{g1}={a:7.3f} | "
            f"{g2}={b:7.3f}"
        )

print()
print("=" * 72)
print("V16 STEP 9N COMPLETE")
print("=" * 72)
print()
print("Send me the output above.")
print("We will compare the fresh TWO geometry against")
print("the original live calibration geometry.")
print("=" * 72)

input("Press Enter to return to PowerShell...")
