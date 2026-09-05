import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

HAND_MODEL_PATH = "models/hand_landmarker.task"

OUTPUT_PATH = "dataset/session_based_evaluation.csv"

MODEL_PATH = "trained_model/gesture_svm_augmented.pkl"
SCALER_PATH = "trained_model/scaler_augmented.pkl"

SAMPLES_PER_GESTURE = 100

# Time between saved samples.
# This prevents collecting almost-identical consecutive frames.
FRAME_SKIP = 5

# Maximum collection time for each gesture.
COLLECTION_SECONDS = 10

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]

SESSION_ID = time.strftime(
    "%Y%m%d_%H%M%S"
)


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" SESSION-BASED GESTURE EVALUATION")
print("==========================================")

print("\nThis program collects a NEW evaluation")
print("session and tests your EXISTING model.")

print("\nIMPORTANT:")
print("The existing model will NOT be modified.")

print(f"\nSession ID: {SESSION_ID}")


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(HAND_MODEL_PATH):

    print("\nERROR: MediaPipe model not found.")
    print(HAND_MODEL_PATH)

    raise SystemExit


if not os.path.exists(MODEL_PATH):

    print("\nERROR: Trained model not found.")
    print(MODEL_PATH)

    raise SystemExit


if not os.path.exists(SCALER_PATH):

    print("\nERROR: Scaler not found.")
    print(SCALER_PATH)

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading existing augmented SVM...")

model = joblib.load(
    MODEL_PATH
)

print("✓ Model loaded.")


print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

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
    f"\nNumber of features: "
    f"{len(feature_names)}"
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

print("\nLoading MediaPipe...")

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


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
    HandLandmarker.create_from_options(
        options
    )
)


print("✓ MediaPipe loaded.")


# ============================================================
# CAMERA
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    print("\nERROR: Could not open webcam.")

    landmarker.close()

    raise SystemExit


print("✓ Webcam opened.")


# ============================================================
# STORAGE
# ============================================================

all_samples = []


# ============================================================
# COLLECT ONE GESTURE
# ============================================================

def collect_gesture(gesture_name):

    samples = []

    frame_counter = 0

    start_time = None

    collecting = False

    print("\n")
    print("==========================================")
    print(
        f" GESTURE: {gesture_name.upper()}"
    )
    print("==========================================")

    print(
        f"\nPrepare the '{gesture_name}' gesture."
    )

    print(
        "Use a slightly different position from"
    )

    print(
        "the previous recordings if possible."
    )

    print(
        "\nYou can move the hand slightly."
    )

    print(
        "Do NOT deliberately change the gesture."
    )

    print(
        "\nPress SPACE to start."
    )

    print(
        "Press Q to quit."
    )


    # ========================================================
    # WAIT FOR SPACE
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:

            continue


        frame = cv2.flip(
            frame,
            1
        )


        cv2.putText(

            frame,

            f"Prepare: {gesture_name}",

            (20, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "Press SPACE to start",

            (20, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "Q = Quit",

            (20, 125),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        cv2.imshow(
            "Session-Based Evaluation",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            return None


        if key == 32:

            collecting = True

            start_time = time.time()

            frame_counter = 0

            break


    # ========================================================
    # COLLECTION
    # ========================================================

    print(
        f"\nCollecting {gesture_name}..."
    )

    print(
        f"Target samples: "
        f"{SAMPLES_PER_GESTURE}"
    )

    print(
        f"Maximum time: "
        f"{COLLECTION_SECONDS} seconds"
    )

    print(
        "Move the hand slightly and naturally."
    )


    while collecting:

        success, frame = cap.read()


        if not success:

            continue


        frame = cv2.flip(
            frame,
            1
        )


        elapsed = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # STOP CONDITIONS
        # ----------------------------------------------------

        if (

            len(samples)
            >= SAMPLES_PER_GESTURE

        ):

            break


        if (

            elapsed
            >= COLLECTION_SECONDS

        ):

            break


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )


        frame_counter += 1


        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        if not result.hand_landmarks:

            cv2.putText(

                frame,

                "NO HAND",

                (20, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 0, 255),

                2
            )


            cv2.putText(

                frame,

                f"{gesture_name}: "
                f"{len(samples)}/"
                f"{SAMPLES_PER_GESTURE}",

                (20, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


            cv2.imshow(

                "Session-Based Evaluation",

                frame
            )


            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                return None


            continue


        # ----------------------------------------------------
        # FRAME SKIPPING
        # ----------------------------------------------------

        if frame_counter % FRAME_SKIP != 0:

            cv2.putText(

                frame,

                "Collecting...",

                (20, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2
            )


            cv2.imshow(

                "Session-Based Evaluation",

                frame
            )


            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                return None


            continue


        # ----------------------------------------------------
        # GET FIRST HAND
        # ----------------------------------------------------

        hand = (
            result.hand_landmarks[0]
        )


        # ----------------------------------------------------
        # EXTRACT 63 FEATURES
        # ----------------------------------------------------

        features = []


        for landmark in hand:

            features.append(
                landmark.x
            )

            features.append(
                landmark.y
            )

            features.append(
                landmark.z
            )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        if len(features) != 63:

            continue


        # ----------------------------------------------------
        # SAVE SAMPLE
        # ----------------------------------------------------

        all_samples.append(
            features
            + [
                gesture_name,
                SESSION_ID
            ]
        )


        samples.append(
            features
        )


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

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

                4,

                (255, 0, 0),

                -1
            )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        remaining = max(

            0,

            COLLECTION_SECONDS
            - elapsed

        )


        cv2.putText(

            frame,

            f"Gesture: {gesture_name}",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Samples: "
            f"{len(samples)}/"
            f"{SAMPLES_PER_GESTURE}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            f"Time: {remaining:.1f}s",

            (20, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "Move hand slightly",

            (20, 150),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "Q = Quit",

            (20, 185),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        cv2.imshow(

            "Session-Based Evaluation",

            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            return None


    # ========================================================
    # RESULT
    # ========================================================

    print(
        f"\n✓ {gesture_name}: "
        f"{len(samples)} samples collected."
    )


    return samples


# ============================================================
# COLLECT ALL GESTURES
# ============================================================

collection_stopped = False


for gesture in GESTURES:

    samples = collect_gesture(
        gesture
    )


    if samples is None:

        print(
            "\nCollection stopped by user."
        )

        collection_stopped = True

        break


    print(
        "\nNext gesture..."
    )

    time.sleep(1)


# ============================================================
# CLEANUP CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


# ============================================================
# CHECK COLLECTION
# ============================================================

if len(all_samples) == 0:

    print(
        "\nNo samples were collected."
    )

    raise SystemExit


# ============================================================
# CREATE DATAFRAME
# ============================================================

columns = (

    feature_names
    + [
        "label",
        "session_id"
    ]

)


df = pd.DataFrame(

    all_samples,

    columns=columns

)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs(

    "dataset",

    exist_ok=True

)


df.to_csv(

    OUTPUT_PATH,

    index=False

)


# ============================================================
# COLLECTION SUMMARY
# ============================================================

print("\n==========================================")
print(" COLLECTION COMPLETE")
print("==========================================")

print(
    f"\nSession ID: {SESSION_ID}"
)

print(
    f"Total samples: {len(df)}"
)

print(
    f"Total features: {len(feature_names)}"
)

print(
    f"Output: {OUTPUT_PATH}"
)


print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = df[
    feature_names
].duplicated().sum()


print("\n==========================================")
print(" DATA QUALITY")
print("==========================================")

print(
    f"\nDuplicate feature rows: {duplicates}"
)


missing_values = (

    df[
        feature_names
    ]
    .isnull()
    .sum()
    .sum()

)


print(
    f"Missing feature values: "
    f"{missing_values}"
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n==========================================")
print(" EXISTING MODEL EVALUATION")
print("==========================================")

print(
    "\nTesting the existing augmented SVM"
)

print(
    "on this completely new session."
)


X_session = df[
    feature_names
]

y_session = df[
    "label"
]


# ------------------------------------------------------------
# SCALE
# ------------------------------------------------------------

X_session_scaled = scaler.transform(
    X_session
)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

predictions = model.predict(
    X_session_scaled
)


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

accuracy = (

    (
        predictions
        == y_session.values
    ).mean()
    * 100

)


print(
    f"\nSession accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# PER-GESTURE ACCURACY
# ============================================================

print("\n==========================================")
print(" PER-GESTURE ACCURACY")
print("==========================================")


for gesture in GESTURES:

    mask = (
        y_session.values
        == gesture
    )


    if mask.sum() == 0:

        continue


    gesture_accuracy = (

        (
            predictions[mask]
            == gesture
        ).mean()
        * 100

    )


    print(

        f"{gesture:12s}: "
        f"{gesture_accuracy:.2f}% "
        f"({mask.sum()} samples)"

    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print(" CONFUSION MATRIX")
print("==========================================")


from sklearn.metrics import confusion_matrix


cm = confusion_matrix(

    y_session,

    predictions,

    labels=GESTURES

)


print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted\n"
)


cm_df = pd.DataFrame(

    cm,

    index=[
        f"Actual_{g}"
        for g in GESTURES
    ],

    columns=[
        f"Pred_{g}"
        for g in GESTURES
    ]

)


print(
    cm_df
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n==========================================")
print(" PREDICTION DISTRIBUTION")
print("==========================================")


prediction_counts = (

    pd.Series(predictions)
    .value_counts()
)


for gesture in GESTURES:

    count = prediction_counts.get(
        gesture,
        0
    )


    percentage = (

        count
        / len(predictions)
        * 100

    )


    print(

        f"{gesture:12s}: "
        f"{count:3d} "
        f"({percentage:.2f}%)"

    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n==========================================")
print(" INTERPRETATION")
print("==========================================")


if accuracy >= 85:

    print(
        "\n✓ EXCELLENT GENERALIZATION"
    )

    print(
        "The existing model performs well"
    )

    print(
        "on a completely new recording session."
    )


elif accuracy >= 70:

    print(
        "\n✓ GOOD GENERALIZATION"
    )

    print(
        "The model generalizes reasonably well,"
    )

    print(
        "but additional robustness testing"
    )

    print(
        "would be useful."
    )


elif accuracy >= 50:

    print(
        "\n⚠ MODERATE GENERALIZATION"
    )

    print(
        "The model works on some unseen"
    )

    print(
        "conditions but still has a domain"
    )

    print(
        "generalization problem."
    )


else:

    print(
        "\n❌ POOR GENERALIZATION"
    )

    print(
        "The model performs poorly on a"
    )

    print(
        "new recording session."
    )

    print(
        "\nThis indicates that the original"
    )

    print(
        "random train/test accuracy is not"
    )

    print(
        "representative of real-world use."
    )


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print(" SESSION EVALUATION FINISHED")
print("==========================================")

print(
    f"\nFinal session accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"\nSaved dataset:"
)

print(
    f"  {OUTPUT_PATH}"
)

print(
    "\nExisting model was NOT modified."
)

print(
    "\n==========================================")