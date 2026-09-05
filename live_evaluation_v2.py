import os
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "trained_model" / "gesture_svm_v2.pkl"
SCALER_PATH = BASE_DIR / "trained_model" / "scaler_v2.pkl"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = DATASET_DIR / "live_evaluation_v2.csv"
OUTPUT_RESULTS = DATASET_DIR / "live_evaluation_v2_results.txt"


# ============================================================
# SETTINGS
# ============================================================

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]

SAMPLES_PER_GESTURE = 100

FRAME_INTERVAL = 5

CAMERA_INDEX = 0

MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


# ============================================================
# FEATURE COLUMN NAMES
# ============================================================

FEATURE_COLUMNS = []

for i in range(21):
    FEATURE_COLUMNS.append(f"x_{i}")
    FEATURE_COLUMNS.append(f"y_{i}")
    FEATURE_COLUMNS.append(f"z_{i}")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(hand_landmarks):
    """
    Extract the same 63 raw x/y/z MediaPipe landmark
    features used by the V2 training dataset.
    """

    features = []

    for landmark in hand_landmarks:

        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)

    features = np.array(
        features,
        dtype=np.float32
    )

    if len(features) != 63:
        return None

    if not np.all(np.isfinite(features)):
        return None

    return features


# ============================================================
# CREATE DATAFRAME FOR SCALER
# ============================================================

def features_to_dataframe(features):
    """
    Convert one 63-element feature vector into a pandas
    DataFrame with the exact feature names used during training.

    This prevents the sklearn feature-name warning.
    """

    return pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

def create_landmarker():

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                HAND_MODEL_PATH
            )
        ),

        running_mode=VisionRunningMode.IMAGE,

        num_hands=1,

        min_hand_detection_confidence=(
            MIN_DETECTION_CONFIDENCE
        ),

        min_hand_presence_confidence=(
            MIN_PRESENCE_CONFIDENCE
        ),

        min_tracking_confidence=(
            MIN_TRACKING_CONFIDENCE
        )
    )

    return HandLandmarker.create_from_options(
        options
    )


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks
):

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    height, width, _ = frame.shape

    points = []

    for landmark in hand_landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    for start, end in connections:

        if (
            start < len(points)
            and end < len(points)
        ):

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )


# ============================================================
# MODEL CHECK
# ============================================================

def check_files():

    print(
        "\n" + "=" * 75
    )

    print(
        "AI HAND GESTURE RECOGNITION - "
        "V2 LIVE EVALUATION"
    )

    print(
        "=" * 75
    )

    print(
        "\n[1] MODEL CHECK"
    )

    print(
        "\nModel:"
    )

    print(
        MODEL_PATH
    )

    if not MODEL_PATH.exists():

        print(
            "\nERROR: V2 model not found!"
        )

        print(
            MODEL_PATH
        )

        return False

    print(
        "OK: V2 model found."
    )

    print(
        "\nScaler:"
    )

    print(
        SCALER_PATH
    )

    if not SCALER_PATH.exists():

        print(
            "\nERROR: V2 scaler not found!"
        )

        print(
            SCALER_PATH
        )

        return False

    print(
        "OK: V2 scaler found."
    )

    print(
        "\nMediaPipe model:"
    )

    print(
        HAND_MODEL_PATH
    )

    if not HAND_MODEL_PATH.exists():

        print(
            "\nERROR: MediaPipe hand model not found!"
        )

        print(
            HAND_MODEL_PATH
        )

        return False

    print(
        "OK: MediaPipe model found."
    )

    return True


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\n[2] LOADING MODEL"
    )

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "SVM model loaded successfully."
    )

    print(
        "Scaler loaded successfully."
    )

    return model, scaler


# ============================================================
# CAMERA TEST
# ============================================================

def open_camera():

    print(
        "\n[3] CAMERA CHECK"
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return None

    print(
        "Camera opened successfully."
    )

    return camera


# ============================================================
# PREDICT ONE FRAME
# ============================================================

def predict_gesture(
    features,
    model,
    scaler
):

    feature_df = features_to_dataframe(
        features
    )

    scaled_features = scaler.transform(
        feature_df
    )

    prediction = model.predict(
        scaled_features
    )[0]

    probabilities = model.predict_proba(
        scaled_features
    )[0]

    classes = list(
        model.classes_
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = classes[
        predicted_index
    ]

    max_probability = float(
        np.max(probabilities)
    )

    return (
        predicted_label,
        max_probability,
        probabilities
    )


# ============================================================
# COLLECT ONE GESTURE
# ============================================================

def collect_gesture_samples(
    camera,
    landmarker,
    model,
    scaler,
    gesture,
    session_id
):

    print(
        "\n" + "-" * 75
    )

    print(
        f"GESTURE: {gesture.upper()}"
    )

    print(
        "-" * 75
    )

    print(
        f"Prepare the gesture '{gesture}' "
        "and press SPACE to start collecting."
    )

    print(
        "Press Q to quit."
    )

    started = False

    samples = []

    frame_counter = 0

    prediction_history = []

    start_time = None

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "WARNING: Could not read camera frame."
            )

            continue

        frame = cv2.flip(
            frame,
            1
        )

        display_frame = frame.copy()

        height, width, _ = (
            display_frame.shape
        )

        # ====================================================
        # WAITING SCREEN
        # ====================================================

        if not started:

            cv2.putText(
                display_frame,
                f"GET READY: {gesture.upper()}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Press SPACE to START",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                "Press Q to QUIT",
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "V2 Live Evaluation",
                display_frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                return None

            if key == 32:

                started = True

                start_time = time.time()

                print(
                    f"\nCollecting "
                    f"{SAMPLES_PER_GESTURE} "
                    f"samples for "
                    f"'{gesture}'..."
                )

            continue

        # ====================================================
        # MEDIAPIPE
        # ====================================================

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

        result = landmarker.detect(
            mp_image
        )

        hand_detected = False

        features = None

        if result.hand_landmarks:

            hand_detected = True

            hand_landmarks = (
                result.hand_landmarks[0]
            )

            draw_hand_landmarks(
                display_frame,
                hand_landmarks
            )

            features = extract_features(
                hand_landmarks
            )

        # ====================================================
        # FRAME INTERVAL
        # ====================================================

        frame_counter += 1

        if (
            hand_detected
            and features is not None
            and frame_counter % FRAME_INTERVAL == 0
        ):

            try:

                (
                    predicted_label,
                    max_probability,
                    probabilities
                ) = predict_gesture(
                    features,
                    model,
                    scaler
                )

                prediction_history.append(
                    predicted_label
                )

                if len(
                    prediction_history
                ) > 10:

                    prediction_history.pop(0)

                # =================================================
                # SAVE ACTUAL LIVE FEATURE VECTOR
                # =================================================

                row = list(
                    features
                )

                row.append(
                    gesture
                )

                row.append(
                    session_id
                )

                samples.append(
                    row
                )

                # =================================================
                # DISPLAY EXPECTED
                # =================================================

                cv2.putText(
                    display_frame,
                    f"Expected: {gesture}",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                # =================================================
                # DISPLAY PREDICTED
                # =================================================

                cv2.putText(
                    display_frame,
                    f"Predicted: {predicted_label}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # =================================================
                # DISPLAY CONFIDENCE
                # =================================================

                cv2.putText(
                    display_frame,
                    f"Confidence: "
                    f"{max_probability * 100:.1f}%",
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            except Exception as error:

                cv2.putText(
                    display_frame,
                    "Prediction error",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                print(
                    f"Prediction error: "
                    f"{error}"
                )

        # ====================================================
        # PROGRESS
        # ====================================================

        progress = (
            f"Samples: "
            f"{len(samples)}/"
            f"{SAMPLES_PER_GESTURE}"
        )

        cv2.putText(
            display_frame,
            progress,
            (30, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # ====================================================
        # TIME
        # ====================================================

        elapsed = (
            time.time()
            - start_time
        )

        cv2.putText(
            display_frame,
            f"Time: {elapsed:.1f}s",
            (30, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "V2 Live Evaluation",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            return None

        # ====================================================
        # TARGET REACHED
        # ====================================================

        if len(samples) >= SAMPLES_PER_GESTURE:

            print(
                f"SUCCESS: Collected "
                f"{len(samples)} samples "
                f"for '{gesture}'."
            )

            break

    return samples


# ============================================================
# SAVE LIVE DATASET
# ============================================================

def save_dataset(all_rows):

    print(
        "\n[4] SAVING LIVE DATASET"
    )

    columns = (
        FEATURE_COLUMNS
        + [
            "label",
            "session_id"
        ]
    )

    dataframe = pd.DataFrame(
        all_rows,
        columns=columns
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"Saved {len(dataframe)} samples."
    )

    print(
        f"File: {OUTPUT_CSV}"
    )

    return dataframe


# ============================================================
# EVALUATION
# ============================================================

def evaluate_dataset(
    dataframe,
    model,
    scaler
):

    print(
        "\n[5] LIVE MODEL EVALUATION"
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith(
            ("x_", "y_", "z_")
        )
    ]

    # ========================================================
    # VERIFY FEATURE COUNT
    # ========================================================

    if len(feature_columns) != 63:

        print(
            "\nERROR: Expected 63 features."
        )

        print(
            f"Found: {len(feature_columns)}"
        )

        return 0.0

    # ========================================================
    # INPUT FEATURES
    # ========================================================

    X = dataframe[
        feature_columns
    ]

    y_true = dataframe[
        "label"
    ]

    # ========================================================
    # SCALE
    # ========================================================

    X_scaled = scaler.transform(
        X
    )

    # ========================================================
    # PREDICT
    # ========================================================

    y_pred = model.predict(
        X_scaled
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "LIVE EVALUATION RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nOverall Live Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=GESTURES,
        zero_division=0
    )

    print(
        report
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\nConfusion Matrix:"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=GESTURES
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{gesture}"
            for gesture in GESTURES
        ],
        columns=[
            f"Pred_{gesture}"
            for gesture in GESTURES
        ]
    )

    print(
        matrix_df
    )

    # ========================================================
    # PER-GESTURE ACCURACY
    # ========================================================

    print(
        "\nPer-Gesture Accuracy:"
    )

    per_gesture = {}

    for gesture in GESTURES:

        mask = (
            y_true == gesture
        )

        total = int(
            mask.sum()
        )

        correct = int(
            (
                (y_pred == gesture)
                & mask
            ).sum()
        )

        if total > 0:

            gesture_accuracy = (
                correct / total
            )

        else:

            gesture_accuracy = 0.0

        per_gesture[
            gesture
        ] = gesture_accuracy

        print(
            f"{gesture:12s}: "
            f"{correct}/{total} "
            f"("
            f"{gesture_accuracy * 100:.2f}%"
            f")"
        )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print(
        "\nPrediction Distribution:"
    )

    prediction_counts = (
        pd.Series(
            y_pred
        ).value_counts()
    )

    for gesture in GESTURES:

        count = int(
            prediction_counts.get(
                gesture,
                0
            )
        )

        percentage = (
            count
            / len(y_pred)
            * 100
        )

        print(
            f"{gesture:12s}: "
            f"{count} "
            f"("
            f"{percentage:.2f}%"
            f")"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AI HAND GESTURE RECOGNITION - "
            "V2 LIVE EVALUATION\n"
        )

        file.write(
            "=" * 75
            + "\n\n"
        )

        file.write(
            f"Total samples: "
            f"{len(dataframe)}\n"
        )

        file.write(
            f"Overall accuracy: "
            f"{accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(
            report
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            matrix_df.to_string()
        )

        file.write(
            "\n\nPer-Gesture Accuracy:\n"
        )

        for gesture in GESTURES:

            file.write(
                f"{gesture}: "
                f"{per_gesture[gesture] * 100:.2f}%\n"
            )

        file.write(
            "\nPrediction Distribution:\n"
        )

        for gesture in GESTURES:

            count = int(
                prediction_counts.get(
                    gesture,
                    0
                )
            )

            file.write(
                f"{gesture}: "
                f"{count}\n"
            )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_RESULTS}"
    )

    return accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not check_files():

        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, scaler = load_model()

    # ========================================================
    # CAMERA
    # ========================================================

    camera = open_camera()

    if camera is None:

        return

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    print(
        "\n[4] INITIALIZING MEDIAPIPE"
    )

    try:

        landmarker = (
            create_landmarker()
        )

        print(
            "MediaPipe Hand Landmarker "
            "initialized."
        )

    except Exception as error:

        print(
            "ERROR: Could not initialize "
            f"MediaPipe: {error}"
        )

        camera.release()

        return

    # ========================================================
    # SESSION ID
    # ========================================================

    session_id = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    print(
        f"\nSession ID: {session_id}"
    )

    print(
        "\nWe will collect "
        f"{SAMPLES_PER_GESTURE} samples "
        "for each of 6 gestures."
    )

    print(
        "\nGestures:"
    )

    for gesture in GESTURES:

        print(
            f"  - {gesture}"
        )

    # ========================================================
    # COLLECT ALL GESTURES
    # ========================================================

    all_rows = []

    for gesture in GESTURES:

        rows = collect_gesture_samples(
            camera,
            landmarker,
            model,
            scaler,
            gesture,
            session_id
        )

        if rows is None:

            print(
                "\nEvaluation cancelled "
                "by user."
            )

            camera.release()

            cv2.destroyAllWindows()

            try:
                landmarker.close()
            except Exception:
                pass

            return

        all_rows.extend(
            rows
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    try:

        landmarker.close()

    except Exception:

        pass

    # ========================================================
    # CHECK DATA
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        f"Total live samples collected: "
        f"{len(all_rows)}"
    )

    if len(all_rows) == 0:

        print(
            "ERROR: No samples were collected."
        )

        return

    # ========================================================
    # SAVE
    # ========================================================

    dataframe = save_dataset(
        all_rows
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    accuracy = evaluate_dataset(
        dataframe,
        model,
        scaler
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "V2 LIVE EVALUATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nFinal Live Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nImportant:"
    )

    print(
        "This result measures performance "
        "on a newly collected webcam session."
    )

    print(
        "It is more meaningful for real-world "
        "generalization than the random "
        "train/test split."
    )

    print(
        f"\nLive dataset:"
        f"\n{OUTPUT_CSV}"
    )

    print(
        f"\nEvaluation report:"
        f"\n{OUTPUT_RESULTS}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()