import cv2
import mediapipe as mp
import numpy as np
import os
import time

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)

DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_calibration",
    "v16_personal_calibration.csv"
)

GESTURES = [
    "one",
    "two",
    "three"
]

MARGIN_THRESHOLD = 1.0
STABLE_FRAMES_REQUIRED = 3


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def normalize(points):

    points = points - points[0]

    scale = np.linalg.norm(points[9])

    if scale < 1e-9:
        return None

    return points / scale


def extract_features(points):

    p = normalize(points)

    if p is None:
        return None

    f = {}

    fingers = {
        "index": (5, 8),
        "middle": (9, 12),
        "ring": (13, 16),
        "pinky": (17, 20)
    }

    for name, (mcp, tip) in fingers.items():

        f[name + "_tip_mcp"] = np.linalg.norm(
            p[tip] - p[mcp]
        )

        f[name + "_tip_wrist"] = np.linalg.norm(
            p[tip]
        )

        f[name + "_ratio"] = (
            np.linalg.norm(p[tip]) /
            (np.linalg.norm(p[mcp]) + 1e-9)
        )

    f["index_middle"] = np.linalg.norm(
        p[8] - p[12]
    )

    f["middle_ring"] = np.linalg.norm(
        p[12] - p[16]
    )

    f["ring_pinky"] = np.linalg.norm(
        p[16] - p[20]
    )

    f["index_ring"] = np.linalg.norm(
        p[8] - p[16]
    )

    f["index_middle_y"] = (
        p[8, 1] - p[12, 1]
    )

    f["ring_pinky_y"] = (
        p[16, 1] - p[20, 1]
    )

    f["ring_pinky_x"] = (
        p[16, 0] - p[20, 0]
    )

    return f


# =========================================================
# LOAD PERSONALIZED CALIBRATION
# =========================================================

print("=" * 70)
print("V16 STEP 9 - REAL-TIME PERSONALIZED RECOGNIZER")
print("=" * 70)

df = __import__("pandas").read_csv(DATASET)

df = df[
    df["label"].isin(GESTURES)
].reset_index(drop=True)

FEATURES = [
    "ring_tip_mcp",
    "index_middle",
    "ring_pinky_y",
    "ring_pinky",
    "ring_ratio",
    "pinky_tip_mcp",
    "middle_tip_mcp",
    "index_tip_mcp",
    "ring_pinky_x",
    "index_middle_y",
    "middle_ring",
    "index_ring"
]

stats = {}

for gesture in GESTURES:

    g = []

    gesture_df = df[
        df["label"] == gesture
    ]

    for _, row in gesture_df.iterrows():

        points = np.array([
            [
                row[f"x_{i}"],
                row[f"y_{i}"],
                row[f"z_{i}"]
            ]
            for i in range(21)
        ], dtype=float)

        f = extract_features(points)

        if f is not None:
            g.append(f)

    stats[gesture] = {}

    for feature in FEATURES:

        values = np.array([
            item[feature]
            for item in g
        ])

        stats[gesture][feature] = (
            values.mean(),
            values.std() + 1e-6
        )

    print(
        f"Loaded {gesture}: {len(g)} samples"
    )


# =========================================================
# PREDICTION
# =========================================================

def predict(feature_dict):

    distances = {}

    for gesture in GESTURES:

        total = 0.0

        for feature in FEATURES:

            mean, std = stats[
                gesture
            ][feature]

            z = (
                feature_dict[feature] - mean
            ) / std

            total += z * z

        distances[gesture] = np.sqrt(
            total / len(FEATURES)
        )

    ordered = sorted(
        distances.items(),
        key=lambda x: x[1]
    )

    best_gesture = ordered[0][0]
    best_distance = ordered[0][1]
    second_distance = ordered[1][1]

    margin = (
        second_distance -
        best_distance
    )

    return (
        best_gesture,
        best_distance,
        margin,
        distances
    )


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(
    options
)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")
    landmarker.close()
    raise SystemExit


print("\nWebcam started.")
print("Press Q to quit.")
print(
    f"Acceptance rule: "
    f"margin >= {MARGIN_THRESHOLD} "
    f"AND {STABLE_FRAMES_REQUIRED} stable frames"
)

previous_prediction = None
stable_count = 0
accepted_gesture = "NONE"

frame_times = []
fps = 0.0


# =========================================================
# REAL-TIME LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    frame = cv2.flip(
        frame,
        1
    )

    start_time = time.perf_counter()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )

    status = "NO HAND"
    gesture_text = "NONE"
    margin_text = "-"
    distance_text = "-"
    stable_text = f"0/{STABLE_FRAMES_REQUIRED}"

    if (
        result.hand_landmarks
        and len(result.hand_landmarks) > 0
    ):

        hand = result.hand_landmarks[0]

        points = np.array([
            [
                lm.x,
                lm.y,
                lm.z
            ]
            for lm in hand
        ], dtype=float)

        feature_dict = extract_features(
            points
        )

        if feature_dict is not None:

            prediction, distance, margin, _ = predict(
                feature_dict
            )

            gesture_text = prediction.upper()
            margin_text = f"{margin:.2f}"
            distance_text = f"{distance:.2f}"

            # -------------------------------------------------
            # Stability tracking
            # -------------------------------------------------

            if (
                prediction ==
                previous_prediction
            ):

                stable_count += 1

            else:

                stable_count = 1
                previous_prediction = prediction

                # Do not immediately retain an old action
                accepted_gesture = "NONE"


            stable_text = (
                f"{stable_count}/"
                f"{STABLE_FRAMES_REQUIRED}"
            )

            # -------------------------------------------------
            # Confidence + temporal acceptance
            # -------------------------------------------------

            if (
                margin >= MARGIN_THRESHOLD
                and
                stable_count >= STABLE_FRAMES_REQUIRED
            ):

                accepted_gesture = prediction

                status = "ACCEPTED"

            elif margin < MARGIN_THRESHOLD:

                status = "HOLD - LOW CONFIDENCE"

            else:

                status = "HOLD - STABILIZING"


    # =====================================================
    # FPS
    # =====================================================

    elapsed = (
        time.perf_counter() -
        start_time
    )

    frame_times.append(
        elapsed
    )

    if len(frame_times) > 30:

        frame_times.pop(0)

    if len(frame_times) > 0:

        fps = 1.0 / (
            sum(frame_times) /
            len(frame_times)
        )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (620, 190),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Accepted: {accepted_gesture.upper()}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (25, 111),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Margin: {margin_text}",
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Distance: {distance_text}",
        (190, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Stable: {stable_text}",
        (350, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (500, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (25, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "V16 Personalized Gesture Recognizer",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\n" + "=" * 70)
print("V16 STEP 9 COMPLETE")
print("=" * 70)
print("\nReal-time recognizer closed.")
