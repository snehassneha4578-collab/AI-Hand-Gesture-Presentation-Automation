import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

HAND_MODEL_PATH = "models/hand_landmarker.task"

OUTPUT_PATH = "dataset/live_diagnostic.csv"

SAMPLES_PER_GESTURE = 100

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" LIVE FEATURE DIAGNOSTIC")
print("==========================================")

print("\nThis program collects live MediaPipe")
print("landmark features for comparison with")
print("your existing training dataset.")

print("\nNo model will be trained or modified.")


# ============================================================
# CREATE DATASET DIRECTORY
# ============================================================

os.makedirs(
    "dataset",
    exist_ok=True
)


# ============================================================
# MEDIAPIPE SETUP
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
# FEATURE NAMES
# ============================================================

feature_names = []

for i in range(21):

    feature_names.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


# ============================================================
# STORAGE
# ============================================================

all_samples = []


# ============================================================
# COLLECT ONE GESTURE
# ============================================================

def collect_gesture(
    gesture_name,
    sample_number,
    total_samples
):

    samples = []

    print("\n")
    print("==========================================")
    print(
        f"GESTURE: {gesture_name.upper()}"
    )
    print("==========================================")

    print(
        f"\nPrepare the '{gesture_name}' gesture."
    )

    print(
        "Hold it naturally in front of the webcam."
    )

    print(
        "\nPress SPACE to start collecting."
    )

    print(
        "Press Q to quit."
    )


    # --------------------------------------------------------
    # WAIT FOR SPACE
    # --------------------------------------------------------

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
            "Live Feature Diagnostic",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            return None


        if key == 32:

            break


    # --------------------------------------------------------
    # COLLECTION
    # --------------------------------------------------------

    print(
        f"\nCollecting {SAMPLES_PER_GESTURE} samples..."
    )

    print(
        "Keep the gesture steady."
    )


    while len(samples) < SAMPLES_PER_GESTURE:

        success, frame = cap.read()


        if not success:

            continue


        frame = cv2.flip(
            frame,
            1
        )


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )


        result = landmarker.detect(
            mp_image
        )


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
                f"{len(samples)}/{SAMPLES_PER_GESTURE}",

                (20, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


            cv2.imshow(
                "Live Feature Diagnostic",
                frame
            )


            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                return None


            continue


        # ----------------------------------------------------
        # GET HAND
        # ----------------------------------------------------

        hand = result.hand_landmarks[0]


        # ----------------------------------------------------
        # EXTRACT FEATURES
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

        cv2.putText(

            frame,

            f"Collecting: {gesture_name}",

            (20, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Samples: "
            f"{len(samples)}/{SAMPLES_PER_GESTURE}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "Keep hand steady",

            (20, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        cv2.imshow(
            "Live Feature Diagnostic",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            return None


    print(
        f"✓ Collected {len(samples)} samples "
        f"for {gesture_name}."
    )


    return samples


# ============================================================
# COLLECT ALL GESTURES
# ============================================================

for gesture in GESTURES:

    samples = collect_gesture(
        gesture,
        0,
        SAMPLES_PER_GESTURE
    )


    if samples is None:

        print("\nCollection stopped by user.")

        break


    # --------------------------------------------------------
    # ADD LABEL
    # --------------------------------------------------------

    for sample in samples:

        row = sample + [gesture]

        all_samples.append(row)


    # --------------------------------------------------------
    # SHORT PAUSE
    # --------------------------------------------------------

    print(
        "\nNext gesture will begin shortly..."
    )

    time.sleep(1)


# ============================================================
# RELEASE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


# ============================================================
# CHECK DATA
# ============================================================

if len(all_samples) == 0:

    print(
        "\nNo samples were collected."
    )

    raise SystemExit


# ============================================================
# CREATE DATAFRAME
# ============================================================

columns = feature_names + ["label"]


df = pd.DataFrame(
    all_samples,
    columns=columns
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print(" COLLECTION COMPLETE")
print("==========================================")

print(
    f"\nTotal samples: {len(df)}"
)

print(
    f"Total features: {len(feature_names)}"
)

print(
    f"Output file: {OUTPUT_PATH}"
)


print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n==========================================")
print(" FEATURE STATISTICS")
print("==========================================")


feature_data = df[
    feature_names
]


# ------------------------------------------------------------
# MINIMUM
# ------------------------------------------------------------

overall_min = (
    feature_data.min().min()
)


# ------------------------------------------------------------
# MAXIMUM
# ------------------------------------------------------------

overall_max = (
    feature_data.max().max()
)


# ------------------------------------------------------------
# MEAN
# ------------------------------------------------------------

overall_mean = (
    feature_data.mean().mean()
)


# ------------------------------------------------------------
# STANDARD DEVIATION
# ------------------------------------------------------------

overall_std = (
    feature_data.std().mean()
)


print(
    f"\nOverall minimum: {overall_min:.6f}"
)

print(
    f"Overall maximum: {overall_max:.6f}"
)

print(
    f"Overall mean: {overall_mean:.6f}"
)

print(
    f"Average feature std: {overall_std:.6f}"
)


# ============================================================
# PER-GESTURE STATISTICS
# ============================================================

print("\n==========================================")
print(" PER-GESTURE STATISTICS")
print("==========================================")


for gesture in GESTURES:

    gesture_data = df[
        df["label"] == gesture
    ][feature_names]


    if len(gesture_data) == 0:

        continue


    mean_value = (
        gesture_data.mean().mean()
    )


    std_value = (
        gesture_data.std().mean()
    )


    min_value = (
        gesture_data.min().min()
    )


    max_value = (
        gesture_data.max().max()
    )


    print(
        f"\n{gesture.upper()}"
    )

    print(
        f"  Samples : {len(gesture_data)}"
    )

    print(
        f"  Mean    : {mean_value:.6f}"
    )

    print(
        f"  Std     : {std_value:.6f}"
    )

    print(
        f"  Min     : {min_value:.6f}"
    )

    print(
        f"  Max     : {max_value:.6f}"
    )


# ============================================================
# DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()


print("\n==========================================")
print(" DUPLICATE CHECK")
print("==========================================")


print(
    f"Duplicate rows: {duplicates}"
)


# ============================================================
# MISSING VALUES
# ============================================================

missing_values = (
    df.isnull().sum().sum()
)


print(
    f"Missing values: {missing_values}"
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print(" DIAGNOSTIC DATA READY")
print("==========================================")

print(
    "\nSaved:"
)

print(
    f"  {OUTPUT_PATH}"
)

print(
    "\nDo NOT retrain the model yet."
)

print(
    "We will compare this live data with"
)

print(
    "your existing training dataset first."
)

print(
    "\n==========================================")