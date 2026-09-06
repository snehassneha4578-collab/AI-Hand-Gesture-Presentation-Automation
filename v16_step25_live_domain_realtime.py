import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

CAL = os.path.join(
    BASE,
    "dataset",
    "v16_calibration",
    "v16_live_calibration_one_two_three.csv"
)

REPORT = os.path.join(
    BASE,
    "dataset",
    "v16_step25_live_domain_realtime_results.txt"
)

TASK = os.path.join(
    BASE,
    "models",
    "hand_landmarker.task"
)

print("="*70)
print("STEP 25 — LIVE-DOMAIN REAL-TIME RECOGNIZER")
print("="*70)

# ---------------------------------------------------------
# LOAD LIVE CALIBRATION
# ---------------------------------------------------------

df = pd.read_csv(CAL)

cols = [
    f"{a}_{i}"
    for i in range(21)
    for a in ["x","y","z"]
]

X = df[cols].values.astype(float)
y = df["label"].values

X = X.reshape(-1,21,3)

print("")
print("Calibration samples:",len(X))

# ---------------------------------------------------------
# DIRECTION FEATURES
# ---------------------------------------------------------

fingers = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]
]

def direction_features(X):

    output = []

    for sample in X:

        features = []

        for finger in fingers:

            pts = sample[finger]

            for j in range(3):

                v = pts[j+1] - pts[j]

                norm = np.linalg.norm(v) + 1e-8

                features.extend([
                    v[0]/norm,
                    v[1]/norm,
                    v[2]/norm
                ])

        output.append(features)

    return np.asarray(output)

D = direction_features(X)

classes = ["one","two","three"]

# ---------------------------------------------------------
# PURE LIVE CENTROIDS
# ---------------------------------------------------------

centroids = {}

for label in classes:

    centroids[label] = D[y == label].mean(axis=0)

print("Live-domain centroids created.")

# ---------------------------------------------------------
# MEDIAPIPE VIDEO MODE
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=TASK
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    landmarker.close()
    raise SystemExit

print("")
print("="*70)
print("REAL-TIME TEST")
print("="*70)
print("")
print("Hold ONE, TWO and THREE naturally.")
print("")
print("The system requires:")
print("  • stable prediction for 3 frames")
print("  • confidence >= 0.60")
print("  • neutral hand state before another command")
print("  • 1 second cooldown")
print("")
print("IMPORTANT:")
print("This test DOES NOT control PowerPoint.")
print("")
print("Press Q to quit.")
print("="*70)

# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------

STABLE_FRAMES = 3
MIN_CONFIDENCE = 0.60
COOLDOWN = 1.0
NEUTRAL_FRAMES = 8

candidate = None
candidate_count = 0

locked = False
neutral_count = 0

last_command = None
last_command_time = 0

accepted = []

frames = 0
hand_frames = 0

start = time.time()
timestamp_ms = 0

# ---------------------------------------------------------
# LOOP
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error.")
        break

    frames += 1

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms += 33

    result = landmarker.detect_for_video(
        image,
        timestamp_ms
    )

    prediction = "NO_HAND"
    confidence = 0.0
    margin = 0.0

    if result.hand_landmarks:

        hand_frames += 1

        landmarks = result.hand_landmarks[0]

        coords = np.array(
            [[p.x,p.y,p.z] for p in landmarks],
            dtype=float
        )

        feat = direction_features(
            coords.reshape(1,21,3)
        )[0]

        # -------------------------------------------------
        # DISTANCE TO LIVE CENTROIDS
        # -------------------------------------------------

        distances = {}

        for label in classes:

            distances[label] = np.linalg.norm(
                feat - centroids[label]
            )

        ordered = sorted(
            distances.items(),
            key=lambda item:item[1]
        )

        best_label = ordered[0][0]
        best_distance = ordered[0][1]

        second_distance = ordered[1][1]

        # Distance-based confidence
        confidence = (
            second_distance /
            (best_distance + second_distance + 1e-8)
        )

        margin = (
            second_distance -
            best_distance
        ) / (
            second_distance +
            best_distance +
            1e-8
        )

        prediction = best_label

        # Draw landmarks
        h,w = frame.shape[:2]

        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]

        x1 = max(0,int(min(xs)*w))
        y1 = max(0,int(min(ys)*h))
        x2 = min(w-1,int(max(xs)*w))
        y2 = min(h-1,int(max(ys)*h))

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        for p in landmarks:

            px = int(p.x*w)
            py = int(p.y*h)

            if 0 <= px < w and 0 <= py < h:

                cv2.circle(
                    frame,
                    (px,py),
                    3,
                    (0,255,255),
                    -1
                )

        # -------------------------------------------------
        # STABILITY
        # -------------------------------------------------

        if (
            confidence >= MIN_CONFIDENCE
        ):

            if prediction == candidate:

                candidate_count += 1

            else:

                candidate = prediction
                candidate_count = 1

            neutral_count = 0

        else:

            candidate = None
            candidate_count = 0
            neutral_count += 1

    else:

        prediction = "NO_HAND"

        confidence = 0.0
        margin = 0.0

        candidate = None
        candidate_count = 0

        neutral_count += 1

    # -----------------------------------------------------
    # UNLOCK
    # -----------------------------------------------------

    if locked and neutral_count >= NEUTRAL_FRAMES:

        locked = False

    # -----------------------------------------------------
    # ACCEPT COMMAND
    # -----------------------------------------------------

    now = time.time()

    if (
        not locked
        and candidate_count >= STABLE_FRAMES
        and confidence >= MIN_CONFIDENCE
        and now-last_command_time >= COOLDOWN
    ):

        command = candidate

        accepted.append({
            "time":now-start,
            "command":command,
            "confidence":confidence,
            "margin":margin
        })

        print(
            f"ACCEPTED: {command.upper()} "
            f"| confidence={confidence:.2f} "
            f"| margin={margin:.2f}"
        )

        last_command = command
        last_command_time = now

        locked = True
        candidate = None
        candidate_count = 0

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    elapsed = time.time()-start

    fps = frames/max(elapsed,1e-8)

    visibility = (
        hand_frames/frames*100
        if frames else 0
    )

    cv2.putText(
        frame,
        f"PREDICTION: {prediction.upper()}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"CONFIDENCE: {confidence:.2f}",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"MARGIN: {margin:.2f}",
        (20,105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"STABLE: {candidate_count}/{STABLE_FRAMES}",
        (20,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"LOCKED: {'YES' if locked else 'NO'}",
        (20,175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20,210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Visibility: {visibility:.1f}%",
        (20,245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Accepted: {len(accepted)}",
        (20,280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Q = quit",
        (20,315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow(
        "STEP 25 - Live Domain Recognizer",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

elapsed = time.time()-start

fps = frames/max(elapsed,1e-8)

visibility = (
    hand_frames/frames*100
    if frames else 0
)

print("")
print("="*70)
print("STEP 25 RESULTS")
print("="*70)

print(f"Frames processed : {frames}")
print(f"Hand frames      : {hand_frames}")
print(f"Hand visibility  : {visibility:.2f}%")
print(f"Average FPS      : {fps:.2f}")
print(f"Accepted commands: {len(accepted)}")

if accepted:

    print("")
    print("Accepted commands:")

    for item in accepted:

        print(
            f"{item['time']:.2f}s | "
            f"{item['command'].upper()} | "
            f"confidence={item['confidence']:.2f} | "
            f"margin={item['margin']:.2f}"
        )

# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

with open(REPORT,"w",encoding="utf-8") as f:

    f.write(
        "STEP 25 — LIVE-DOMAIN REAL-TIME RECOGNIZER\n"
    )

    f.write("="*70+"\n\n")

    f.write(
        f"Frames processed: {frames}\n"
    )

    f.write(
        f"Hand frames: {hand_frames}\n"
    )

    f.write(
        f"Hand visibility: {visibility:.2f}%\n"
    )

    f.write(
        f"Average FPS: {fps:.2f}\n"
    )

    f.write(
        f"Accepted commands: {len(accepted)}\n\n"
    )

    for item in accepted:

        f.write(
            f"{item['time']:.2f}s | "
            f"{item['command']} | "
            f"confidence={item['confidence']:.4f} | "
            f"margin={item['margin']:.4f}\n"
        )

print("")
print("Report saved:")
print(REPORT)

print("")
print("="*70)
print("STEP 25 COMPLETE")
print("="*70)
