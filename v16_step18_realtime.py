import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque, Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
CAL = os.path.join(BASE, "dataset", "v16_calibration", "v16_live_calibration_one_two_three.csv")
MODEL_OUT = os.path.join(BASE, "trained_model", "v16_step18_direction_knn.pkl")
REPORT = os.path.join(BASE, "dataset", "v16_step18_realtime_results.txt")
TASK = os.path.join(BASE, "models", "hand_landmarker.task")

# ---------------------------------------------------------
# LOAD CALIBRATION DATA
# ---------------------------------------------------------

df = pd.read_csv(CAL)

feature_cols = [f"{a}_{i}" for i in range(21) for a in ["x","y","z"]]
X_raw = df[feature_cols].values.astype(float)
y = np.asarray(df["label"].values)

X_raw = X_raw.reshape(-1, 21, 3)

fingers = {
    "thumb":  [1,2,3,4],
    "index":  [5,6,7,8],
    "middle": [9,10,11,12],
    "ring":   [13,14,15,16],
    "pinky":  [17,18,19,20]
}

def make_direction_features(X):
    output = []

    for sample in X:
        features = []

        for name in ["thumb","index","middle","ring","pinky"]:
            pts = sample[fingers[name]]

            for j in range(3):
                v = pts[j+1] - pts[j]
                n = np.linalg.norm(v) + 1e-8
                v = v / n
                features.extend(v.tolist())

        output.append(features)

    return np.asarray(output, dtype=float)

X = make_direction_features(X_raw)

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

model.fit(X_scaled, y)

# ---------------------------------------------------------
# MEDIAPIPE
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

# ---------------------------------------------------------
# TEMPORAL SETTINGS
# ---------------------------------------------------------

history = deque(maxlen=5)

STABLE_FRAMES = 3
MIN_CONFIDENCE = 0.55
COOLDOWN = 1.5

last_command = "NONE"
last_accept_time = 0

frame_count = 0
hand_frames = 0
accepted = 0
start_time = time.time()

print("\n" + "="*70)
print("STEP 18 — REAL-TIME CONFIDENCE + TEMPORAL STABILITY")
print("="*70)
print("Controls:")
print("  q = quit")
print("  c = clear history")
print("\nMove your hand naturally.")
print("A gesture is accepted only after temporal stability + confidence.")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    prediction = "UNKNOWN"
    confidence = 0.0
    accepted_now = False

    if result.hand_landmarks:

        hand_frames += 1

        landmarks = result.hand_landmarks[0]

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=float
        )

        live_raw = coords.reshape(1,21,3)

        live_features = make_direction_features(live_raw)
        live_scaled = scaler.transform(live_features)

        probabilities = model.predict_proba(live_scaled)[0]
        classes = model.classes_

        order = np.argsort(probabilities)[::-1]

        best_idx = order[0]
        second_idx = order[1]

        prediction = classes[best_idx]

        best_prob = probabilities[best_idx]
        second_prob = probabilities[second_idx]

        # Combine probability and margin
        margin = best_prob - second_prob

        confidence = 0.6 * best_prob + 0.4 * margin

        # -------------------------------------------------
        # REJECTION
        # -------------------------------------------------

        if confidence >= MIN_CONFIDENCE:
            history.append(prediction)
        else:
            history.append("UNKNOWN")

        # -------------------------------------------------
        # TEMPORAL STABILITY
        # -------------------------------------------------

        if len(history) >= STABLE_FRAMES:

            recent = list(history)[-STABLE_FRAMES:]

            if (
                recent.count(recent[-1]) == STABLE_FRAMES
                and recent[-1] != "UNKNOWN"
            ):

                stable_prediction = recent[-1]

                now = time.time()

                if (
                    stable_prediction != last_command
                    or now - last_accept_time >= COOLDOWN
                ):

                    accepted_now = True
                    accepted += 1

                    last_command = stable_prediction
                    last_accept_time = now

                    print(
                        f"ACCEPTED: {stable_prediction.upper()} "
                        f"| confidence={confidence:.2f}"
                    )

    else:

        history.append("UNKNOWN")

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    cv2.putText(
        frame,
        f"Prediction: {prediction.upper()}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Stable: {list(history)[-STABLE_FRAMES:] if len(history)>=STABLE_FRAMES else list(history)}",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Last Accepted: {last_command.upper()}",
        (20,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20,180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    if accepted_now:
        cv2.putText(
            frame,
            "COMMAND ACCEPTED",
            (20,225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            3
        )

    cv2.imshow("Step 18 - Real-Time Gesture Engine", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        history.clear()
        print("History cleared.")

cap.release()
cv2.destroyAllWindows()
landmarker.close()

elapsed = time.time() - start_time

print("\n" + "="*70)
print("STEP 18 SESSION SUMMARY")
print("="*70)

print(f"Frames processed : {frame_count}")
print(f"Hand frames      : {hand_frames}")
print(f"Hand coverage    : {hand_frames/frame_count*100:.2f}%")
print(f"Average FPS      : {frame_count/elapsed:.2f}")
print(f"Accepted commands: {accepted}")
print(f"Last command     : {last_command}")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("STEP 18 — REAL-TIME CONFIDENCE + TEMPORAL STABILITY\n")
    f.write("="*70 + "\n\n")
    f.write(f"Frames processed: {frame_count}\n")
    f.write(f"Hand frames: {hand_frames}\n")
    f.write(f"Hand coverage: {hand_frames/frame_count*100:.2f}%\n")
    f.write(f"Average FPS: {frame_count/elapsed:.2f}\n")
    f.write(f"Accepted commands: {accepted}\n")
    f.write(f"Last command: {last_command}\n")
    f.write(f"Stable frames: {STABLE_FRAMES}\n")
    f.write(f"Minimum confidence: {MIN_CONFIDENCE}\n")
    f.write(f"Cooldown: {COOLDOWN} seconds\n")

print(f"\nReport saved: {REPORT}")
