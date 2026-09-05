import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
from collections import deque, Counter


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "trained_model/gesture_svm_augmented.pkl"
SCALER_PATH = "trained_model/scaler_augmented.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"

CAMERA_INDEX = 0

# Normal prediction filtering
CONFIDENCE_THRESHOLD = 55.0
MIN_MARGIN = 10.0

# Temporal stabilization
HISTORY_SIZE = 7
MIN_STABLE_COUNT = 4
MIN_STABILITY = 60.0

# Very strong prediction shortcut
HIGH_CONFIDENCE = 90.0
HIGH_MARGIN = 40.0
HIGH_CONFIDENCE_FRAMES = 3

# Time required before generating an event
MIN_HOLD_TIME = 0.20

# Tolerance for occasional weak frames
MAX_UNCERTAIN_FRAMES = 3


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" AI HAND GESTURE RECOGNITION")
print(" SMART TEMPORAL STABILIZATION")
print("==========================================")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading augmented SVM...")

model = joblib.load(MODEL_PATH)

print("✓ Augmented SVM loaded.")


print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("✓ Scaler loaded.")


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = []

for i in range(21):

    feature_names.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


print(
    f"\n✓ Number of features: {len(feature_names)}"
)


# ============================================================
# MEDIAPIPE
# ============================================================

print("\nLoading MediaPipe...")

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=RunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.3,

    min_hand_presence_confidence=0.3,

    min_tracking_confidence=0.3
)


landmarker = (
    HandLandmarker.create_from_options(options)
)


print("✓ MediaPipe loaded.")


# ============================================================
# CAMERA
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(CAMERA_INDEX)


if not cap.isOpened():

    print("\nERROR: Webcam could not be opened.")

    landmarker.close()

    raise SystemExit


print("✓ Webcam opened.")


# ============================================================
# TEMPORAL STATE
# ============================================================

prediction_history = deque(
    maxlen=HISTORY_SIZE
)

confidence_history = deque(
    maxlen=HISTORY_SIZE
)


# Strong prediction counter
high_confidence_history = deque(
    maxlen=HIGH_CONFIDENCE_FRAMES
)


# Current candidate
candidate_gesture = None

candidate_start_time = None


# Last generated event
last_event_gesture = None


# Current stable gesture
current_stable_gesture = None


# Number of weak frames
uncertain_frames = 0


# ============================================================
# FPS
# ============================================================

previous_time = time.time()

fps = 0.0


# ============================================================
# EVENT COUNT
# ============================================================

event_count = 0


# ============================================================
# START
# ============================================================

print("\n==========================================")
print(" REAL-TIME TEST STARTED")
print("==========================================")

print("\nGestures:")

print("  fist")
print("  one")
print("  two")
print("  three")
print("  open_palm")
print("  pinch")

print("\nHold each gesture for 3–5 seconds.")

print("\nPress Q to quit.\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    success, frame = cap.read()


    if not success:

        print("[LIVE] Could not read webcam frame.")

        continue


    # ========================================================
    # MIRROR
    # ========================================================

    frame = cv2.flip(frame, 1)


    # ========================================================
    # RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    # ========================================================
    # DETECT HAND
    # ========================================================

    result = landmarker.detect(mp_image)


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    if not result.hand_landmarks:

        uncertain_frames += 1


        print("[LIVE] No hand detected.")


        # Do NOT immediately destroy the stable
        # gesture because of a short detection drop.

        if uncertain_frames > MAX_UNCERTAIN_FRAMES:

            prediction_history.clear()

            confidence_history.clear()

            high_confidence_history.clear()

            candidate_gesture = None

            candidate_start_time = None

            current_stable_gesture = None

            last_event_gesture = None


        cv2.putText(

            frame,

            "No hand detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


        cv2.putText(

            frame,

            f"Events: {event_count}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        cv2.imshow(
            "AI Hand Gesture Recognition",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        continue


    # ========================================================
    # HAND FOUND
    # ========================================================

    uncertain_frames = 0


    hand = result.hand_landmarks[0]


    # ========================================================
    # EXTRACT 21 LANDMARKS
    # ========================================================

    features = []


    for landmark in hand:

        features.append(landmark.x)

        features.append(landmark.y)

        features.append(landmark.z)


    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    if len(features) != 63:

        print("[ERROR] Invalid feature count.")

        continue


    # ========================================================
    # DATAFRAME
    # ========================================================

    X_live = pd.DataFrame(

        [features],

        columns=feature_names
    )


    # ========================================================
    # SCALE
    # ========================================================

    X_scaled = scaler.transform(
        X_live.values
    )


    # ========================================================
    # PREDICT
    # ========================================================

    probabilities = model.predict_proba(
        X_scaled
    )[0]


    classes = model.classes_


    # ========================================================
    # SORT PREDICTIONS
    # ========================================================

    sorted_indices = np.argsort(
        probabilities
    )[::-1]


    top_index = sorted_indices[0]

    second_index = sorted_indices[1]


    top_gesture = classes[top_index]

    second_gesture = classes[second_index]


    top_confidence = (

        probabilities[top_index] * 100
    )


    second_confidence = (

        probabilities[second_index] * 100
    )


    # ========================================================
    # MARGIN
    # ========================================================

    margin = (

        top_confidence
        - second_confidence
    )


    # ========================================================
    # GOOD PREDICTION
    # ========================================================

    prediction_is_good = (

        top_confidence >= CONFIDENCE_THRESHOLD

        and

        margin >= MIN_MARGIN
    )


    # ========================================================
    # HIGH-CONFIDENCE PREDICTION
    # ========================================================

    very_strong_prediction = (

        top_confidence >= HIGH_CONFIDENCE

        and

        margin >= HIGH_MARGIN
    )


    # ========================================================
    # ACCEPT PREDICTION
    # ========================================================

    if prediction_is_good:

        prediction_history.append(
            top_gesture
        )

        confidence_history.append(
            top_confidence
        )


    # ========================================================
    # HIGH-CONFIDENCE HISTORY
    # ========================================================

    if very_strong_prediction:

        high_confidence_history.append(
            top_gesture
        )

    else:

        high_confidence_history.clear()


    # ========================================================
    # STABILIZATION
    # ========================================================

    stable_gesture = None

    stability = 0.0

    stable_confidence = 0.0

    majority_count = 0


    # ========================================================
    # METHOD 1:
    # HIGH-CONFIDENCE FAST STABILIZATION
    # ========================================================

    if len(high_confidence_history) >= HIGH_CONFIDENCE_FRAMES:

        high_counter = Counter(
            high_confidence_history
        )


        high_gesture, high_count = (
            high_counter.most_common(1)[0]
        )


        if high_count >= HIGH_CONFIDENCE_FRAMES:

            stable_gesture = high_gesture

            stability = 100.0

            stable_confidence = top_confidence


    # ========================================================
    # METHOD 2:
    # NORMAL TEMPORAL VOTING
    # ========================================================

    if stable_gesture is None:

        if len(prediction_history) > 0:

            counter = Counter(
                prediction_history
            )


            majority_gesture, majority_count = (

                counter.most_common(1)[0]
            )


            stability = (

                majority_count
                / len(prediction_history)
            ) * 100


            majority_confidences = []


            for i in range(
                len(prediction_history)
            ):

                if (

                    prediction_history[i]
                    == majority_gesture

                ):

                    majority_confidences.append(
                        confidence_history[i]
                    )


            if majority_confidences:

                stable_confidence = (

                    sum(majority_confidences)
                    /
                    len(majority_confidences)
                )


            if (

                majority_count
                >= MIN_STABLE_COUNT

                and

                stability
                >= MIN_STABILITY

            ):

                stable_gesture = (
                    majority_gesture
                )


    # ========================================================
    # CANDIDATE GESTURE
    # ========================================================

    if stable_gesture is not None:

        if candidate_gesture != stable_gesture:

            candidate_gesture = stable_gesture

            candidate_start_time = time.time()


        hold_time = (

            time.time()
            - candidate_start_time
        )


        # ====================================================
        # GENERATE EVENT
        # ====================================================

        if hold_time >= MIN_HOLD_TIME:

            if candidate_gesture != last_event_gesture:

                last_event_gesture = candidate_gesture

                current_stable_gesture = candidate_gesture

                event_count += 1


                print(
                    "\n>>> GESTURE EVENT: "
                    f"{candidate_gesture.upper()}"
                )


    # ========================================================
    # NO STABLE GESTURE
    # ========================================================

    else:

        candidate_gesture = None

        candidate_start_time = None


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()


    elapsed = (
        current_time
        - previous_time
    )


    if elapsed > 0:

        instant_fps = 1.0 / elapsed


        if fps == 0:

            fps = instant_fps

        else:

            fps = (
                0.9 * fps
                +
                0.1 * instant_fps
            )


    previous_time = current_time


    # ========================================================
    # DISPLAY STABLE STATE
    # ========================================================

    if stable_gesture is not None:

        stable_text = stable_gesture

    else:

        stable_text = "uncertain"


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(

        f"[LIVE] "

        f"Top: {top_gesture} "
        f"{top_confidence:.1f}% | "

        f"Second: {second_gesture} "
        f"{second_confidence:.1f}% | "

        f"Margin: {margin:.1f}% | "

        f"Stable: {stable_text} | "

        f"Stability: {stability:.1f}% | "

        f"FPS: {fps:.1f}"

    )


    # ========================================================
    # CAMERA TEXT
    # ========================================================

    cv2.putText(

        frame,

        f"Top: {top_gesture}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 255, 0),

        2
    )


    cv2.putText(

        frame,

        f"Confidence: {top_confidence:.1f}%",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"Margin: {margin:.1f}%",

        (20, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"Stable: {stable_text}",

        (20, 145),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"Stability: {stability:.1f}%",

        (20, 180),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (20, 215),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"Events: {event_count}",

        (20, 250),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    # ========================================================
    # EVENT MESSAGE
    # ========================================================

    if current_stable_gesture is not None:

        cv2.putText(

            frame,

            f"ACTIVE: {current_stable_gesture.upper()}",

            (20, 300),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0, 255, 0),

            3
        )


    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    for landmark in hand:

        x = int(
            landmark.x
            * frame.shape[1]
        )

        y = int(
            landmark.y
            * frame.shape[0]
        )


        cv2.circle(

            frame,

            (x, y),

            5,

            (255, 0, 0),

            -1
        )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(

        "AI Hand Gesture Recognition",

        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print("\n==========================================")
print(" REAL-TIME TEST FINISHED")
print("==========================================")

print(
    f"Total gesture events: {event_count}"
)

print("==========================================")