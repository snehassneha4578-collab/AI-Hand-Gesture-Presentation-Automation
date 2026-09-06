import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

ROOT = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
CSV = ROOT / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
MODEL = ROOT / "models" / "hand_landmarker.task"
REPORT = ROOT / "dataset" / "v16_step28_controlled_live_accuracy.txt"

print("="*70)
print("STEP 28 — CONTROLLED REAL-TIME ACCURACY TEST")
print("="*70)

df = pd.read_csv(CSV)

def direction_features(row):
    pts = np.array([
        [row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]]
        for i in range(21)
    ], dtype=np.float32)

    fingers = {
        "thumb": [1,2,3,4],
        "index": [5,6,7,8],
        "middle": [9,10,11,12],
        "ring": [13,14,15,16],
        "pinky": [17,18,19,20]
    }

    feats = []

    for ids in fingers.values():
        for a, b in zip(ids[:-1], ids[1:]):
            v = pts[b] - pts[a]
            n = np.linalg.norm(v) + 1e-8
            feats.extend((v / n).tolist())

    return np.array(feats, dtype=np.float32)

def hand_to_features(hand):
    row = {
        f"{axis}_{i}": getattr(p, axis)
        for i, p in enumerate(hand)
        for axis in ["x", "y", "z"]
    }
    return direction_features(pd.Series(row))

X = np.array([
    direction_features(r)
    for _, r in df.iterrows()
])

centroids = {
    label: X[df["label"].values == label].mean(axis=0)
    for label in ["one", "two", "three"]
}

print("\nLive-domain centroids loaded.")
print("\nTEST PROCEDURE")
print("-"*70)
print("For each gesture:")
print("  1. Hold the gesture naturally.")
print("  2. Press SPACE once.")
print("  3. Keep holding until the result appears.")
print("  4. Relax your hand before the next attempt.")
print("")
print("10 attempts per gesture.")
print("")
print("Order:")
print("  ONE   -> 10 attempts")
print("  TWO   -> 10 attempts")
print("  THREE -> 10 attempts")
print("")
print("Press Q to quit.")
print("="*70)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

gesture_order = ["one", "two", "three"]

attempts = {
    "one": [],
    "two": [],
    "three": []
}

current_gesture_index = 0
attempt_number = 0
space_was_down = False

stable_prediction = None
stable_count = 0

frame_count = 0
hand_count = 0
start = time.time()

last_prediction = "NO HAND"
last_margin = 0.0
last_best_distance = 0.0

with HandLandmarker.create_from_options(options) as landmarker:

    while current_gesture_index < len(gesture_order):

        ok, frame = cap.read()

        if not ok:
            continue

        frame_count += 1

        expected = gesture_order[current_gesture_index]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = int(time.time() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        prediction = None
        margin = 0.0
        best_distance = 0.0

        if result.hand_landmarks:

            hand_count += 1

            f = hand_to_features(result.hand_landmarks[0])

            distances = {
                label: float(
                    np.linalg.norm(f - centroid)
                )
                for label, centroid in centroids.items()
            }

            ordered = sorted(
                distances.items(),
                key=lambda x: x[1]
            )

            prediction = ordered[0][0]
            best_distance = ordered[0][1]
            second_distance = ordered[1][1]

            margin = second_distance - best_distance

            if prediction == stable_prediction:
                stable_count += 1
            else:
                stable_prediction = prediction
                stable_count = 1

            last_prediction = prediction.upper()
            last_margin = margin
            last_best_distance = best_distance

        else:

            stable_prediction = None
            stable_count = 0
            last_prediction = "NO HAND"
            last_margin = 0.0

        cv2.putText(
            frame,
            f"TARGET: {expected.upper()}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"PREDICTION: {last_prediction}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"MARGIN: {last_margin:.3f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"STABLE: {stable_count}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"ATTEMPT: {len(attempts[expected]) + 1}/10",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            "SPACE = record attempt | Q = quit",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2
        )

        cv2.imshow(
            "Step 28 - Controlled Accuracy Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Detect a new SPACE press.
        if key == 32 and not space_was_down:

            if prediction is not None:

                attempts[expected].append({
                    "prediction": prediction,
                    "margin": margin,
                    "best_distance": best_distance,
                    "stable_frames": stable_count
                })

                attempt_number += 1

                correct = prediction == expected

                print(
                    f"Attempt {len(attempts[expected]):2d}/10 | "
                    f"TRUE={expected.upper():5s} | "
                    f"PRED={prediction.upper():5s} | "
                    f"{'CORRECT' if correct else 'WRONG':7s} | "
                    f"margin={margin:.3f} | "
                    f"stable={stable_count}"
                )

            else:

                attempts[expected].append({
                    "prediction": "no_hand",
                    "margin": 0.0,
                    "best_distance": 0.0,
                    "stable_frames": 0
                })

                attempt_number += 1

                print(
                    f"Attempt {len(attempts[expected]):2d}/10 | "
                    f"TRUE={expected.upper():5s} | "
                    f"PRED=NO_HAND | WRONG"
                )

            if len(attempts[expected]) >= 10:

                if current_gesture_index < len(gesture_order) - 1:

                    current_gesture_index += 1

                    print("\n" + "-"*70)
                    print(
                        f"NEXT TARGET: "
                        f"{gesture_order[current_gesture_index].upper()}"
                    )
                    print("-"*70 + "\n")

                else:

                    print("\nAll 30 attempts completed.")

        space_was_down = (key == 32)

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("STEP 28 RESULTS")
print("="*70)

total = 0
correct_total = 0

report_lines = []

report_lines.append("="*70)
report_lines.append("STEP 28 — CONTROLLED REAL-TIME ACCURACY TEST")
report_lines.append("="*70)
report_lines.append("")

for gesture in gesture_order:

    data = attempts[gesture]

    total_g = len(data)
    correct_g = sum(
        x["prediction"] == gesture
        for x in data
    )

    accuracy = (
        correct_g / total_g * 100
        if total_g else 0
    )

    total += total_g
    correct_total += correct_g

    margins = [
        x["margin"]
        for x in data
        if x["prediction"] != "no_hand"
    ]

    report_lines.append(
        f"{gesture.upper():5s}: "
        f"{correct_g}/{total_g} = {accuracy:.2f}%"
    )

    if margins:

        report_lines.append(
            f"       margin mean={np.mean(margins):.3f}, "
            f"median={np.median(margins):.3f}, "
            f"max={np.max(margins):.3f}"
        )

overall = (
    correct_total / total * 100
    if total else 0
)

report_lines.append("")
report_lines.append(
    f"OVERALL: {correct_total}/{total} = {overall:.2f}%"
)

report_lines.append("")
report_lines.append("No PowerPoint control was used.")
report_lines.append(
    "This is a controlled real-time recognition measurement."
)

text = "\n".join(report_lines)

print(text)

REPORT.write_text(
    text,
    encoding="utf-8"
)

print("\nReport saved:")
print(REPORT)

print("\n" + "="*70)
print("STEP 28 COMPLETE")
print("="*70)
