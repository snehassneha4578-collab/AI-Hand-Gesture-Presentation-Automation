import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL = BASE + r"\trained_model\v16_specialized_live_engine.pkl"
TASK = BASE + r"\models\hand_landmarker.task"

with open(MODEL, "rb") as f:
    engine = pickle.load(f)

FEATURES = engine["features"]
STATS = engine["stats"]
WEIGHTS = engine["weights"]
GESTURES = ["one", "two", "three"]

def make_features(pts):
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

        f[f"{name}_tip_wrist"] = np.linalg.norm(pts[idx])

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

def predict(hand):
    pts = np.array(
        [[p.x, p.y, p.z] for p in hand],
        dtype=np.float32
    )

    f = make_features(pts)
    x = np.array([f[name] for name in FEATURES], dtype=np.float32)

    distances = {}

    for gesture in GESTURES:
        mean = STATS[gesture]["mean"]
        std = STATS[gesture]["std"]

        z = (x - mean) / std

        # Weighted standardized distance
        distances[gesture] = np.sqrt(
            np.sum(WEIGHTS * z * z)
        )

    ranked = sorted(
        distances.items(),
        key=lambda x: x[1]
    )

    best = ranked[0][0]
    second = ranked[1][0]

    margin = ranked[1][1] - ranked[0][1]

    return best, margin, distances

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK),
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
print("V16 STEP 9M - SPECIALIZED LIVE ENGINE TEST")
print("=" * 72)
print("Show ONE, TWO and THREE separately.")
print("Observe which gesture wins.")
print("Press Q to quit.")
print("=" * 72)

counts = {g: 0 for g in GESTURES}
total = 0

start = time.time()

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

        if result.hand_landmarks:

            best, margin, distances = predict(
                result.hand_landmarks[0]
            )

            total += 1
            counts[best] += 1

            cv2.putText(
                frame,
                f"PREDICTION: {best.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y = 120

            for gesture, distance in sorted(
                distances.items(),
                key=lambda x: x[1]
            ):

                cv2.putText(
                    frame,
                    f"{gesture.upper():5s}: {distance:.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                y += 30

        else:

            cv2.putText(
                frame,
                "NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        fps = total / max(time.time() - start, 1e-6)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "V16 Step 9M - Specialized Engine",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 72)
print("V16 STEP 9M COMPLETE")
print("=" * 72)
print(f"Frames with hand: {total}")

if total:
    for g in GESTURES:
        print(
            f"{g.upper():5s}: "
            f"{counts[g]:4d} "
            f"({counts[g] / total * 100:.2f}%)"
        )

print("=" * 72)

input("Press Enter to return to PowerShell...")
