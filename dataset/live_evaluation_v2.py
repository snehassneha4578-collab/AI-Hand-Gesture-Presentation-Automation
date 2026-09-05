# ============================================================
# AI HAND GESTURE RECOGNITION - V2 LIVE EVALUATION
# ============================================================

from pathlib import Path
import csv
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "trained_model" / "gesture_svm_v2.pkl"
SCALER_PATH = BASE_DIR / "trained_model" / "scaler_v2.pkl"

OUTPUT_PATH = BASE_DIR / "dataset" / "live_evaluation_v2.csv"
RESULT_PATH = BASE_DIR / "dataset" / "live_evaluation_v2_results.txt"


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

MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 75)
print("AI HAND GESTURE RECOGNITION - V2 LIVE EVALUATION")
print("=" * 75)


# ============================================================
# 1. CHECK MODEL
# ============================================================

print("\n[1] MODEL CHECK")

if not MODEL_PATH.exists():
    print("\nERROR: V2 model not found!")
    print(MODEL_PATH)
    raise SystemExit

if not SCALER_PATH.exists():
    print("\nERROR: V2 scaler not found!")
    print(SCALER_PATH)
    raise SystemExit

print("V2 model found:")
print(MODEL_PATH)

print("\nV2 scaler found:")
print(SCALER_PATH)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\n[2] LOADING MODEL")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully.")
print("Scaler loaded successfully.")


# ============================================================
# 3. MEDIAPIPE SETUP
# ============================================================

print("\n[3] INITIALIZING MEDIAPIPE")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(
            BASE_DIR / "models" / "hand_landmarker.task"
        )
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def extract_features(hand_landmarks):
    """
    Extract 63 features:
    21 landmarks × (x, y, z)
    """

    features = []

    for landmark in hand_landmarks:
        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)

    return features


# ============================================================
# 5. CAMERA
# ============================================================

print("\n[4] OPENING CAMERA")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print("Camera opened successfully.")


# ============================================================
# 6. PREPARE OUTPUT
# ============================================================

feature_names = []

for i in range(21):
    feature_names.append(f"x_{i}")
    feature_names.append(f"y_{i}")
    feature_names.append(f"z_{i}")

output_columns = feature_names + [
    "actual_label",
    "predicted_label",
    "confidence"
]

all_results = []


# ============================================================
# 7. START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    for gesture_index, actual_gesture in enumerate(GESTURES):

        print("\n" + "=" * 75)
        print(
            f"GESTURE {gesture_index + 1}/{len(GESTURES)}: "
            f"{actual_gesture.upper()}"
        )
        print("=" * 75)

        print("\nPrepare your hand for:")
        print(f"   {actual_gesture.upper()}")

        print("\nPress SPACE to start collecting.")
        print("Press Q to quit.")

        started = False
        sample_count = 0
        frame_count = 0

        # --------------------------------------------------------
        # Wait for SPACE
        # --------------------------------------------------------

        while not started:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"NEXT GESTURE: {actual_gesture.upper()}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press SPACE to start | Q to quit",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "V2 Live Evaluation",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                started = True

            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        print("\nCollection started.")

        # --------------------------------------------------------
        # Collect samples
        # --------------------------------------------------------

        while sample_count < SAMPLES_PER_GESTURE:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            frame_count += 1

            # Process every Nth frame
            if frame_count % FRAME_INTERVAL != 0:
                cv2.imshow(
                    "V2 Live Evaluation",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

                continue

            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            # ----------------------------------------------------
            # Hand detected
            # ----------------------------------------------------

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                features = extract_features(hand)

                # Validate
                if len(features) == 63 and np.isfinite(features).all():

                    X_live = np.array(
                        features,
                        dtype=np.float32
                    ).reshape(1, -1)

                    # Scale using training scaler
                    X_scaled = scaler.transform(X_live)

                    # Prediction
                    prediction = model.predict(X_scaled)[0]

                    # Probability / confidence
                    probabilities = model.predict_proba(
                        X_scaled
                    )[0]

                    max_index = np.argmax(probabilities)

                    confidence = (
                        probabilities[max_index] * 100
                    )

                    # ------------------------------------------------
                    # Save result
                    # ------------------------------------------------

                    row = features + [
                        actual_gesture,
                        prediction,
                        confidence
                    ]

                    all_results.append(row)

                    sample_count += 1

                    # ------------------------------------------------
                    # Draw landmarks
                    # ------------------------------------------------

                    for landmark in hand:

                        h, w, _ = frame.shape

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                    # ------------------------------------------------
                    # Display prediction
                    # ------------------------------------------------

                    cv2.putText(
                        frame,
                        f"Actual: {actual_gesture}",
                        (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Predicted: {prediction}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.1f}%",
                        (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    print(
                        f"\r{actual_gesture:<12} "
                        f"{sample_count:3d}/{SAMPLES_PER_GESTURE} | "
                        f"Pred: {prediction:<12} | "
                        f"Conf: {confidence:6.2f}%",
                        end=""
                    )

            else:

                cv2.putText(
                    frame,
                    "NO HAND DETECTED",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # --------------------------------------------------------
            # Progress
            # --------------------------------------------------------

            cv2.putText(
                frame,
                f"Samples: {sample_count}/{SAMPLES_PER_GESTURE}",
                (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "V2 Live Evaluation",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        print("\n\nGesture collection completed.")


# ============================================================
# 8. RELEASE CAMERA
# ============================================================

cap.release()
cv2.destroyAllWindows()


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 75)
print("[5] LIVE DATASET CREATED")
print("=" * 75)

results_df = pd.DataFrame(
    all_results,
    columns=output_columns
)

print(f"\nTotal live samples: {len(results_df)}")

print("\nSamples per gesture:")

print(
    results_df["actual_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. SAVE LIVE DATA
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nLive evaluation dataset saved:")
print(OUTPUT_PATH)


# ============================================================
# 11. ACCURACY
# ============================================================

y_true = results_df["actual_label"]
y_pred = results_df["predicted_label"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 75)
print("V2 LIVE ACCURACY")
print("=" * 75)

print(
    f"\nLive Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=GESTURES,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 75)
print("LIVE CLASSIFICATION REPORT")
print("=" * 75)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=GESTURES
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{g}" for g in GESTURES],
    columns=[f"Pred_{g}" for g in GESTURES]
)

print("\n" + "=" * 75)
print("LIVE CONFUSION MATRIX")
print("=" * 75)

print(cm_df)


# ============================================================
# 14. PER-GESTURE ACCURACY
# ============================================================

print("\n" + "=" * 75)
print("PER-GESTURE LIVE ACCURACY")
print("=" * 75)

for gesture in GESTURES:

    gesture_data = results_df[
        results_df["actual_label"] == gesture
    ]

    gesture_accuracy = (
        gesture_data["predicted_label"] == gesture
    ).mean() * 100

    correct = (
        gesture_data["predicted_label"] == gesture
    ).sum()

    total = len(gesture_data)

    print(
        f"{gesture:<15} "
        f"{gesture_accuracy:6.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# 15. CONFIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("CONFIDENCE ANALYSIS")
print("=" * 75)

print(
    f"\nAverage confidence: "
    f"{results_df['confidence'].mean():.2f}%"
)

print(
    f"Minimum confidence: "
    f"{results_df['confidence'].min():.2f}%"
)

print(
    f"Maximum confidence: "
    f"{results_df['confidence'].max():.2f}%"
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 75 + "\n")
    f.write("AI HAND GESTURE RECOGNITION - V2 LIVE EVALUATION\n")
    f.write("=" * 75 + "\n\n")

    f.write("MODEL\n")
    f.write("-" * 75 + "\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Scaler: {SCALER_PATH}\n\n")

    f.write("DATASET\n")
    f.write("-" * 75 + "\n")
    f.write(f"Total live samples: {len(results_df)}\n")
    f.write(f"Samples per gesture: {SAMPLES_PER_GESTURE}\n")
    f.write(f"Number of gestures: {len(GESTURES)}\n\n")

    f.write("LIVE ACCURACY\n")
    f.write("-" * 75 + "\n")
    f.write(
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 75 + "\n")
    f.write(report)
    f.write("\n")

    f.write("CONFUSION MATRIX\n")
    f.write("-" * 75 + "\n")
    f.write(cm_df.to_string())
    f.write("\n\n")

    f.write("PER-GESTURE ACCURACY\n")
    f.write("-" * 75 + "\n")

    for gesture in GESTURES:

        gesture_data = results_df[
            results_df["actual_label"] == gesture
        ]

        gesture_accuracy = (
            gesture_data["predicted_label"] == gesture
        ).mean() * 100

        correct = (
            gesture_data["predicted_label"] == gesture
        ).sum()

        total = len(gesture_data)

        f.write(
            f"{gesture:<15} "
            f"{gesture_accuracy:6.2f}% "
            f"({correct}/{total})\n"
        )

    f.write("\nCONFIDENCE\n")
    f.write("-" * 75 + "\n")
    f.write(
        f"Average: {results_df['confidence'].mean():.2f}%\n"
    )
    f.write(
        f"Minimum: {results_df['confidence'].min():.2f}%\n"
    )
    f.write(
        f"Maximum: {results_df['confidence'].max():.2f}%\n"
    )


# ============================================================
# 17. FINAL
# ============================================================

print("\n" + "=" * 75)
print("V2 LIVE EVALUATION COMPLETE")
print("=" * 75)

print(
    f"\nFINAL LIVE ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print("\nFiles created:")

print(f"  - {OUTPUT_PATH}")
print(f"  - {RESULT_PATH}")

print("\nNEXT STEP:")
print("Analyze live performance and improve the model if required.")

print("\n" + "=" * 75)