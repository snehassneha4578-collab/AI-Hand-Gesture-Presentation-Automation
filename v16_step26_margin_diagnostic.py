import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

ROOT = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
CSV = ROOT / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
MODEL = ROOT / "models" / "hand_landmarker.task"

print("="*70)
print("STEP 26 — REAL-TIME LIVE-DOMAIN MARGIN DIAGNOSTIC")
print("="*70)

df = pd.read_csv(CSV)

FEATURES = []

def direction_features(row):
    pts = np.array([
        [row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]]
        for i in range(21)
    ], dtype=np.float32)

    wrist = pts[0]
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

for _, r in df.iterrows():
    FEATURES.append(direction_features(r))

X = np.array(FEATURES)

centroids = {}
for label in ["one", "two", "three"]:
    centroids[label] = X[df["label"].values == label].mean(axis=0)

print("\nLive-domain centroids ready.")
print("Hold ONE, TWO and THREE naturally.")
print("Press Q to quit.\n")

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

def landmark_row(hand):
    row = {}
    for i, p in enumerate(hand):
        row[f"x_{i}"] = p.x
        row[f"y_{i}"] = p.y
        row[f"z_{i}"] = p.z
    return row

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
hand_count = 0
start = time.time()

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

        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            hand_count += 1

            row = landmark_row(result.hand_landmarks[0])
            f = direction_features(pd.Series(row))

            distances = {
                label: float(np.linalg.norm(f - centroid))
                for label, centroid in centroids.items()
            }

            ordered = sorted(distances.items(), key=lambda x: x[1])

            best_label, best_distance = ordered[0]
            second_label, second_distance = ordered[1]

            margin = second_distance - best_distance

            # Relative separation score.
            separation = second_distance / (best_distance + second_distance + 1e-8)

            cv2.putText(
                frame,
                f"Prediction: {best_label.upper()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Best distance: {best_distance:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Second distance: {second_distance:.3f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.3f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Separation: {separation:.3f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"ONE {distances['one']:.2f} | TWO {distances['two']:.2f} | THREE {distances['three']:.2f}",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255,255,0),
                2
            )

        fps = frame_count / max(time.time() - start, 1e-8)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Hand: {hand_count/max(frame_count,1)*100:.1f}%",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,255),
            2
        )

        cv2.imshow("Step 26 - Margin Diagnostic", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("STEP 26 COMPLETE")
print("="*70)
print(f"Frames processed : {frame_count}")
print(f"Hand frames      : {hand_count}")
print(f"Hand visibility  : {hand_count/max(frame_count,1)*100:.2f}%")
print(f"Average FPS      : {frame_count/max(time.time()-start,1e-8):.2f}")
print("="*70)
