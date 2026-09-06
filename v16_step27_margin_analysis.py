import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

ROOT = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
CSV = ROOT / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
MODEL = ROOT / "models" / "hand_landmarker.task"
REPORT = ROOT / "dataset" / "v16_step27_margin_analysis.txt"

print("="*70)
print("STEP 27 — AUTOMATIC LIVE MARGIN ANALYSIS")
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

def row_from_hand(hand):
    row = {}
    for i, p in enumerate(hand):
        row[f"x_{i}"] = p.x
        row[f"y_{i}"] = p.y
        row[f"z_{i}"] = p.z
    return pd.Series(row)

X = np.array([direction_features(r) for _, r in df.iterrows()])

centroids = {
    label: X[df["label"].values == label].mean(axis=0)
    for label in ["one", "two", "three"]
}

print("\nLive-domain centroids ready.")
print("Hold ONE for a few seconds, then TWO, then THREE.")
print("Move naturally but keep the hand clearly visible.")
print("Press Q when finished.\n")

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

records = []
frame_count = 0
hand_count = 0
start = time.time()

last_print = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        frame_count += 1

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

        if result.hand_landmarks:

            hand_count += 1

            row = row_from_hand(result.hand_landmarks[0])

            f = direction_features(row)

            distances = {
                label: float(np.linalg.norm(f - centroid))
                for label, centroid in centroids.items()
            }

            ordered = sorted(
                distances.items(),
                key=lambda x: x[1]
            )

            best_label = ordered[0][0]
            best_distance = ordered[0][1]

            second_label = ordered[1][0]
            second_distance = ordered[1][1]

            margin = second_distance - best_distance

            separation = (
                second_distance /
                (best_distance + second_distance + 1e-8)
            )

            records.append({
                "frame": frame_count,
                "prediction": best_label,
                "best_distance": best_distance,
                "second_distance": second_distance,
                "second_label": second_label,
                "margin": margin,
                "separation": separation
            })

            cv2.putText(
                frame,
                f"PREDICTION: {best_label.upper()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Separation: {separation:.3f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"ONE:{distances['one']:.2f} TWO:{distances['two']:.2f} THREE:{distances['three']:.2f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255,255,0),
                2
            )

            now = time.time()

            if now - last_print > 1.0:

                print(
                    f"{best_label.upper():5s} | "
                    f"best={best_distance:.3f} | "
                    f"second={second_distance:.3f} | "
                    f"margin={margin:.3f} | "
                    f"sep={separation:.3f}"
                )

                last_print = now

        fps = frame_count / max(time.time() - start, 1e-8)

        cv2.putText(
            frame,
            f"FPS:{fps:.1f} Hand:{hand_count/max(frame_count,1)*100:.1f}%",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,255),
            2
        )

        cv2.imshow(
            "Step 27 - Automatic Margin Analysis",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

elapsed = max(time.time() - start, 1e-8)

result_df = pd.DataFrame(records)

lines = []

lines.append("="*70)
lines.append("STEP 27 — AUTOMATIC LIVE MARGIN ANALYSIS")
lines.append("="*70)
lines.append("")
lines.append(f"Frames processed : {frame_count}")
lines.append(f"Hand frames      : {hand_count}")
lines.append(
    f"Hand visibility  : {hand_count/max(frame_count,1)*100:.2f}%"
)
lines.append(f"Average FPS      : {frame_count/elapsed:.2f}")
lines.append(f"Prediction records: {len(result_df)}")
lines.append("")

if len(result_df) > 0:

    lines.append("="*70)
    lines.append("PREDICTION DISTRIBUTION")
    lines.append("="*70)

    counts = result_df["prediction"].value_counts()

    for label in ["one", "two", "three"]:
        count = int(counts.get(label, 0))
        pct = count / len(result_df) * 100
        lines.append(
            f"{label.upper():5s}: {count:5d} ({pct:6.2f}%)"
        )

    lines.append("")
    lines.append("="*70)
    lines.append("MARGIN STATISTICS BY PREDICTED CLASS")
    lines.append("="*70)

    for label in ["one", "two", "three"]:

        subset = result_df[
            result_df["prediction"] == label
        ]

        lines.append("")
        lines.append(f"{label.upper()} — samples: {len(subset)}")

        if len(subset) > 0:

            for col in [
                "best_distance",
                "second_distance",
                "margin",
                "separation"
            ]:

                s = subset[col]

                lines.append(
                    f"  {col:17s}: "
                    f"min={s.min():.3f}  "
                    f"mean={s.mean():.3f}  "
                    f"median={s.median():.3f}  "
                    f"max={s.max():.3f}"
                )

    lines.append("")
    lines.append("="*70)
    lines.append("MARGIN THRESHOLD COVERAGE")
    lines.append("="*70)

    for threshold in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60
    ]:

        accepted = result_df[
            result_df["margin"] >= threshold
        ]

        coverage = (
            len(accepted) /
            len(result_df) *
            100
        )

        lines.append(
            f"Margin >= {threshold:.2f}: "
            f"{len(accepted):5d} frames "
            f"({coverage:6.2f}%)"
        )

    lines.append("")
    lines.append("="*70)
    lines.append("INTERPRETATION")
    lines.append("="*70)
    lines.append(
        "This diagnostic measures the actual real-time distance "
        "and margin behavior of the live-domain recognizer."
    )
    lines.append(
        "It is used to tune command acceptance safely."
    )
    lines.append(
        "It does NOT control PowerPoint."
    )

else:

    lines.append("No hand detections were recorded.")

text = "\n".join(lines)

print("\n")
print(text)

REPORT.write_text(
    text,
    encoding="utf-8"
)

print("\nReport saved:")
print(REPORT)

print("\n" + "="*70)
print("STEP 27 COMPLETE")
print("="*70)
