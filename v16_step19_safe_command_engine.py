import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
CAL = os.path.join(BASE, "dataset", "v16_calibration", "v16_live_calibration_one_two_three.csv")
TASK = os.path.join(BASE, "models", "hand_landmarker.task")
REPORT = os.path.join(BASE, "dataset", "v16_step19_command_engine_results.txt")

print("STEP 19 STARTING...")
print("Loading calibration data...")

df = pd.read_csv(CAL)

cols = [f"{a}_{i}" for i in range(21) for a in ["x","y","z"]]

X = df[cols].values.astype(np.float64)
y = np.asarray(df["label"].values)

X = X.reshape(-1,21,3)

fingers = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]
]

def direction_features(data):

    output = []

    for sample in data:

        f = []

        for finger in fingers:

            pts = sample[finger]

            for j in range(3):

                v = pts[j+1] - pts[j]

                norm = np.sqrt(
                    v[0]**2 +
                    v[1]**2 +
                    v[2]**2
                ) + 1e-8

                f.extend([
                    v[0]/norm,
                    v[1]/norm,
                    v[2]/norm
                ])

        output.append(f)

    return np.asarray(output,dtype=np.float64)

features = direction_features(X)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

model.fit(features_scaled,y)

print("Model ready.")

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
    landmarker.close()
    raise SystemExit

# ---------------------------------------------------------
# COMMAND ENGINE
# ---------------------------------------------------------

history = deque(maxlen=5)

STABLE_FRAMES = 3

MIN_CONFIDENCE = 0.60

# Minimum time before another accepted command
COOLDOWN = 1.0

# Number of neutral frames required to unlock
NEUTRAL_FRAMES = 8

last_prediction = "UNKNOWN"

accepted_command = "NONE"

locked = False

neutral_count = 0

last_accept_time = 0

frames = 0
hands = 0
accepted = 0

start = time.time()

print("")
print("="*70)
print("STEP 19 — SAFE COMMAND ENGINE")
print("="*70)
print("")
print("Gesture sequence:")
print("")
print("  HOLD GESTURE")
print("       ↓")
print("  STABLE FRAMES")
print("       ↓")
print("  ACCEPT ONCE")
print("       ↓")
print("  LOCK")
print("       ↓")
print("  RELEASE / CHANGE")
print("       ↓")
print("  NEUTRAL")
print("       ↓")
print("  UNLOCK")
print("")
print("Press Q to quit.")
print("="*70)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    frames += 1

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(image)

    prediction = "UNKNOWN"
    confidence = 0.0

    if result.hand_landmarks:

        hands += 1

        landmarks = result.hand_landmarks[0]

        coords = np.array(
            [[p.x,p.y,p.z] for p in landmarks],
            dtype=np.float64
        )

        live = direction_features(
            coords.reshape(1,21,3)
        )

        live_scaled = scaler.transform(live)

        probs = model.predict_proba(live_scaled)[0]

        order = np.argsort(probs)[::-1]

        best = order[0]
        second = order[1]

        prediction = model.classes_[best]

        p1 = probs[best]
        p2 = probs[second]

        margin = p1-p2

        confidence = 0.6*p1 + 0.4*margin

        # ---------------------------------------------
        # ACCEPTABLE PREDICTION
        # ---------------------------------------------

        if confidence >= MIN_CONFIDENCE:

            history.append(prediction)

        else:

            history.append("UNKNOWN")

        # ---------------------------------------------
        # DETECT NEUTRAL / UNKNOWN
        # ---------------------------------------------

        if prediction == "UNKNOWN" or confidence < MIN_CONFIDENCE:

            neutral_count += 1

        else:

            neutral_count = 0

        # ---------------------------------------------
        # UNLOCK AFTER NEUTRAL
        # ---------------------------------------------

        if neutral_count >= NEUTRAL_FRAMES:

            locked = False
            accepted_command = "READY"

        # ---------------------------------------------
        # STABILITY
        # ---------------------------------------------

        if len(history) >= STABLE_FRAMES:

            recent = list(history)[-STABLE_FRAMES:]

            stable = (
                recent.count(recent[-1]) == STABLE_FRAMES
                and recent[-1] != "UNKNOWN"
            )

            if stable and not locked:

                candidate = recent[-1]

                now = time.time()

                if now-last_accept_time >= COOLDOWN:

                    accepted_command = candidate.upper()

                    last_prediction = candidate

                    last_accept_time = now

                    accepted += 1

                    locked = True

                    neutral_count = 0

                    print(
                        f"ACCEPTED: {candidate.upper()} "
                        f"| confidence={confidence:.2f}"
                    )

    else:

        history.append("UNKNOWN")

        neutral_count += 1

        if neutral_count >= NEUTRAL_FRAMES:

            locked = False
            accepted_command = "READY"

    elapsed = time.time()-start

    fps = frames/elapsed if elapsed > 0 else 0

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Prediction: {prediction.upper()}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    state = "LOCKED" if locked else "READY"

    cv2.putText(
        frame,
        f"Command State: {state}",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Last Command: {accepted_command}",
        (20,145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Neutral Frames: {neutral_count}/{NEUTRAL_FRAMES}",
        (20,180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20,215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow(
        "STEP 19 - Safe Gesture Command Engine",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()

elapsed = time.time()-start

coverage = hands/frames*100 if frames else 0
fps = frames/elapsed if elapsed else 0

print("")
print("="*70)
print("STEP 19 SUMMARY")
print("="*70)
print(f"Frames processed : {frames}")
print(f"Hand frames      : {hands}")
print(f"Hand coverage    : {coverage:.2f}%")
print(f"Average FPS      : {fps:.2f}")
print(f"Accepted commands: {accepted}")
print(f"Last command     : {accepted_command}")
print("="*70)

with open(REPORT,"w",encoding="utf-8") as f:

    f.write("STEP 19 — SAFE COMMAND ENGINE\n")
    f.write("="*70+"\n\n")
    f.write(f"Frames processed: {frames}\n")
    f.write(f"Hand frames: {hands}\n")
    f.write(f"Hand coverage: {coverage:.2f}%\n")
    f.write(f"Average FPS: {fps:.2f}\n")
    f.write(f"Accepted commands: {accepted}\n")
    f.write(f"Last command: {accepted_command}\n")
    f.write(f"Stable frames: {STABLE_FRAMES}\n")
    f.write(f"Minimum confidence: {MIN_CONFIDENCE}\n")
    f.write(f"Neutral frames required: {NEUTRAL_FRAMES}\n")
    f.write(f"Cooldown: {COOLDOWN}s\n")

print(f"\nReport saved: {REPORT}")
