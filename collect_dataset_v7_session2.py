from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import uuid

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
DATASET_PATH = BASE_DIR / "dataset" / "gesture_dataset_v7.csv"

GESTURES = [
    ("fist", "Close all fingers"),
    ("one", "Index finger only"),
    ("two", "Index + middle fingers"),
    ("three", "Index + middle + ring fingers"),
    ("open_palm", "All five fingers open"),
    ("pinch", "Thumb + index fingertip touching")
]

TARGET_SAMPLES = 350
MAX_SECONDS = 120
FRAME_INTERVAL = 6
MIN_MOVEMENT = 0.004

FEATURE_COLUMNS = [
    f"{axis}_{i}"
    for i in range(21)
    for axis in ["x", "y", "z"]
]

COLUMNS = FEATURE_COLUMNS + ["label", "session_id"]

# ------------------------------------------------------------
# Validate existing V7 dataset
# ------------------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"V7 dataset not found:\n{DATASET_PATH}"
    )

df_existing = pd.read_csv(DATASET_PATH)

if list(df_existing.columns) != COLUMNS:
    raise ValueError(
        "Existing V7 dataset has an unexpected structure."
    )

print("=" * 75)
print("V7 SESSION 2 DATA COLLECTION")
print("=" * 75)

print(f"\nExisting rows: {len(df_existing)}")
print(f"Existing sessions: {df_existing['session_id'].nunique()}")

print("\nExisting session IDs:")

for sid in df_existing["session_id"].unique():
    label = df_existing.loc[
        df_existing["session_id"] == sid,
        "label"
    ].iloc[0]

    print(f"  {label:12} {sid}")

print("""
IMPORTANT:

You are now collecting a SECOND INDEPENDENT SESSION.

Do NOT try to reproduce Session 1.

For every gesture:
- Change your hand position.
- Change distance from camera.
- Slightly change rotation/tilt.
- Move left/right/up/down.
- Use natural small movements.
- Keep the gesture clearly formed.
- Keep the complete hand visible.
- Avoid rapidly changing gestures.

This session is specifically for testing
UNSEEN-SESSION GENERALIZATION.
""")

input(
    "\nPress ENTER to begin Session 2 collection..."
)

# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_landmarks(result):

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    features = []

    for landmark in hand:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.asarray(
        features,
        dtype=np.float32
    )


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


# ------------------------------------------------------------
# Collect Session 2
# ------------------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    for gesture_index, (gesture, description) in enumerate(GESTURES):

        print("\n" + "=" * 75)
        print(f"SESSION 2 → {gesture.upper()}")
        print(f"Instruction: {description}")
        print("=" * 75)

        input(
            "Change your position/distance/orientation, then press ENTER..."
        )

        session_id = (
            time.strftime("%Y%m%d_%H%M%S")
            + "_S2_"
            + uuid.uuid4().hex[:6]
        )

        samples = []

        start_time = time.time()
        frame_count = 0
        last_features = None

        print("\nCollecting...")
        print("Move naturally through different positions.")

        while True:

            elapsed = time.time() - start_time

            if elapsed >= MAX_SECONDS:
                print("\nMaximum session time reached.")
                break

            if len(samples) >= TARGET_SAMPLES:
                print("\nTarget samples reached.")
                break

            ret, frame = cap.read()

            if not ret:
                continue

            frame_count += 1

            frame = cv2.flip(frame, 1)

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

            features = extract_landmarks(result)

            if features is not None:

                accept = False

                if last_features is None:

                    accept = True

                else:

                    movement = np.linalg.norm(
                        features - last_features
                    )

                    if movement >= MIN_MOVEMENT:
                        accept = True

                if (
                    accept
                    and frame_count % FRAME_INTERVAL == 0
                ):

                    samples.append(features)

                    last_features = features.copy()

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            progress = (
                len(samples)
                / TARGET_SAMPLES
            ) * 100

            cv2.putText(
                frame,
                f"SESSION 2 | {gesture}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {len(samples)}/{TARGET_SAMPLES}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Progress: {progress:.1f}%",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Vary position + distance + tilt",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q = stop current gesture",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "V7 Session 2 Collector",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(
                    "\nCurrent gesture stopped manually."
                )
                break

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        if samples:

            session_df = pd.DataFrame(
                samples,
                columns=FEATURE_COLUMNS
            )

            session_df["label"] = gesture
            session_df["session_id"] = session_id

            session_df = session_df[COLUMNS]

            session_df.to_csv(
                DATASET_PATH,
                mode="a",
                header=False,
                index=False
            )

            print(
                f"\nSaved {len(samples)} samples."
            )

            print(
                f"Session ID: {session_id}"
            )

        else:

            print(
                f"\nWARNING: No samples saved for {gesture}."
            )

        if gesture_index < len(GESTURES) - 1:

            input(
                "\nPress ENTER for the next gesture..."
            )


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()


# ------------------------------------------------------------
# Final dataset summary
# ------------------------------------------------------------

final_df = pd.read_csv(DATASET_PATH)

print("\n" + "=" * 75)
print("V7 SESSION 2 COLLECTION COMPLETE")
print("=" * 75)

print(f"\nTotal rows: {len(final_df)}")
print(
    f"Total sessions: "
    f"{final_df['session_id'].nunique()}"
)

print("\nGESTURE COUNTS")

print(
    final_df["label"]
    .value_counts()
    .sort_index()
)

print("\nSESSION COUNTS")

print(
    final_df
    .groupby(["label", "session_id"])
    .size()
    .to_string()
)

print(
    f"\nDataset:\n{DATASET_PATH}"
)

print("\nExpected result:")
print("6 original sessions + 6 new sessions = 12 sessions")
