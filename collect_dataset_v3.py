# ============================================================
# COLLECT DATASET V3
# AI-Based Hand Gesture Recognition & Presentation Automation
#
# Purpose:
# Collect NEW independent sessions for each gesture.
#
# The data is appended safely to:
# dataset/gesture_dataset_v2.csv
#
# Each session:
#   - 1 gesture
#   - 250 target samples
#   - unique session_id
#   - 63 MediaPipe x/y/z features
#
# These new sessions will later be used by train_model_v3.py
# for a true session-aware evaluation.
# ============================================================

from pathlib import Path
import csv
import time
import uuid
import sys

import cv2
import mediapipe as mp


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

DATASET_PATH = DATASET_DIR / "gesture_dataset_v2.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# 2. SETTINGS
# ============================================================

CAMERA_INDEX = 0

TARGET_SAMPLES = 250

FRAME_INTERVAL = 6

MIN_MOVEMENT = 0.008

MAX_SESSION_SECONDS = 90

MIN_DETECTION_CONFIDENCE = 0.30

MIN_PRESENCE_CONFIDENCE = 0.30

MIN_TRACKING_CONFIDENCE = 0.30


# ============================================================
# 3. GESTURES
# ============================================================

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]


# ============================================================
# 4. MEDIAPIPE
# ============================================================

mp_tasks = mp.tasks

mp_vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp_vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp_vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp_vision.RunningMode
)


# ============================================================
# 5. FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = []

for i in range(21):

    FEATURE_COLUMNS.append(
        f"x_{i}"
    )

    FEATURE_COLUMNS.append(
        f"y_{i}"
    )

    FEATURE_COLUMNS.append(
        f"z_{i}"
    )


CSV_COLUMNS = (
    FEATURE_COLUMNS
    +
    [
        "label",
        "session_id"
    ]
)


# ============================================================
# 6. CHECK FILES
# ============================================================

print()
print("=" * 70)
print("AI HAND GESTURE DATASET V3 COLLECTION")
print("=" * 70)

print()

print(
    f"Dataset:\n{DATASET_PATH}"
)

print()

print(
    f"MediaPipe model:\n{MODEL_PATH}"
)


if not MODEL_PATH.exists():

    print()

    print(
        "ERROR: hand_landmarker.task not found."
    )

    print(
        "Expected location:"
    )

    print(
        MODEL_PATH
    )

    sys.exit(1)


# ============================================================
# 7. CREATE DATASET FOLDER
# ============================================================

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 8. CREATE OR VALIDATE CSV
# ============================================================

if not DATASET_PATH.exists():

    print()

    print(
        "Dataset does not exist."
    )

    print(
        "Creating a new dataset..."
    )

    with open(
        DATASET_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            CSV_COLUMNS
        )

else:

    print()

    print(
        "Existing dataset found."
    )

    print(
        "Checking CSV header..."
    )

    try:

        with open(
            DATASET_PATH,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(file)

            existing_header = next(
                reader
            )

    except Exception as e:

        print()

        print(
            "ERROR: Could not read dataset."
        )

        print(
            e
        )

        sys.exit(1)


    if existing_header != CSV_COLUMNS:

        print()

        print(
            "ERROR: Dataset header does not match."
        )

        print()

        print(
            "Expected columns:"
        )

        print(
            len(CSV_COLUMNS)
        )

        print()

        print(
            "Found columns:"
        )

        print(
            len(existing_header)
        )

        print()

        print(
            "DO NOT append data."
        )

        sys.exit(1)


    print(
        "CSV header is valid."
    )


# ============================================================
# 9. HELPER FUNCTION
# ============================================================

def extract_landmark_features(hand):

    features = []

    for landmark in hand:

        features.append(
            float(landmark.x)
        )

        features.append(
            float(landmark.y)
        )

        features.append(
            float(landmark.z)
        )

    return features


# ============================================================
# 10. MOVEMENT CALCULATION
# ============================================================

def calculate_movement(
    current_features,
    previous_features
):

    if previous_features is None:

        return 999.0

    total = 0.0

    count = 0

    for i in range(
        0,
        63,
        3
    ):

        dx = (
            current_features[i]
            -
            previous_features[i]
        )

        dy = (
            current_features[i + 1]
            -
            previous_features[i + 1]
        )

        dz = (
            current_features[i + 2]
            -
            previous_features[i + 2]
        )

        distance = (
            dx * dx
            +
            dy * dy
            +
            dz * dz
        ) ** 0.5

        total += distance

        count += 1

    if count == 0:

        return 0.0

    return total / count


# ============================================================
# 11. COLLECT ONE SESSION
# ============================================================

def collect_session(
    gesture,
    landmarker
):

    print()
    print("=" * 70)
    print(
        f"NEW SESSION: {gesture.upper()}"
    )
    print("=" * 70)

    print()

    print(
        "Before starting, vary:"
    )

    print(
        "  - left / center / right"
    )

    print(
        "  - upper / middle / lower"
    )

    print(
        "  - closer / farther"
    )

    print(
        "  - hand rotation"
    )

    print(
        "  - natural small movements"
    )

    print(
        "  - different orientations"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Keep the SAME gesture shape,"
    )

    print(
        "but vary the hand position and orientation."
    )

    print()

    print(
        "Press SPACE to START."
    )

    print(
        "Press Q to CANCEL this session."
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print()

        print(
            "ERROR: Could not open camera."
        )

        return False


    # --------------------------------------------------------
    # SESSION ID
    # --------------------------------------------------------

    session_id = (
        time.strftime(
            "%Y%m%d_%H%M%S"
        )
        +
        "_"
        +
        uuid.uuid4().hex[:6]
    )


    samples = []

    previous_features = None

    frame_counter = 0

    started = False

    start_time = None

    last_saved_time = 0


    # --------------------------------------------------------
    # CAMERA LOOP
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # ----------------------------------------------------
        # MIRROR DISPLAY
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # CONVERT BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        timestamp_ms = (
            int(
                time.time() * 1000
            )
        )


        try:

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

        except Exception:

            result = None


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        hand_detected = False

        current_features = None


        if (
            result is not None
            and
            result.hand_landmarks
        ):

            hand_detected = True

            hand = (
                result.hand_landmarks[0]
            )


            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            for landmark in hand:

                x = int(
                    landmark.x
                    *
                    frame.shape[1]
                )

                y = int(
                    landmark.y
                    *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # DRAW CONNECTIONS
            # ------------------------------------------------

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


            for a, b in connections:

                x1 = int(
                    hand[a].x
                    *
                    frame.shape[1]
                )

                y1 = int(
                    hand[a].y
                    *
                    frame.shape[0]
                )

                x2 = int(
                    hand[b].x
                    *
                    frame.shape[1]
                )

                y2 = int(
                    hand[b].y
                    *
                    frame.shape[0]
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------------
            # EXTRACT 63 FEATURES
            # ------------------------------------------------

            current_features = (
                extract_landmark_features(
                    hand
                )
            )


        # ====================================================
        # BEFORE START
        # ====================================================

        if not started:

            cv2.putText(
                frame,
                f"GESTURE: {gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "PRESS SPACE TO START",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q = Cancel",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # AFTER START
        # ====================================================

        else:

            elapsed = (
                time.time()
                -
                start_time
            )


            # ------------------------------------------------
            # AUTO STOP
            # ------------------------------------------------

            if (
                elapsed
                >=
                MAX_SESSION_SECONDS
            ):

                print()

                print(
                    "Maximum session time reached."
                )

                break


            # ------------------------------------------------
            # MOVEMENT
            # ------------------------------------------------

            movement = (
                calculate_movement(
                    current_features,
                    previous_features
                )
                if current_features is not None
                else 0.0
            )


            # ------------------------------------------------
            # SAMPLE ACCEPTANCE
            # ------------------------------------------------

            frame_counter += 1


            if (
                current_features is not None
                and
                frame_counter % FRAME_INTERVAL == 0
                and
                movement >= MIN_MOVEMENT
            ):

                # --------------------------------------------
                # Validate finite values
                # --------------------------------------------

                valid = all(
                    (
                        value == value
                        and
                        abs(value) != float("inf")
                    )
                    for value
                    in current_features
                )


                if valid:

                    samples.append(
                        current_features
                    )

                    previous_features = (
                        current_features.copy()
                    )

                    last_saved_time = (
                        time.time()
                    )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            progress = (
                f"{len(samples)}/"
                f"{TARGET_SAMPLES}"
            )

            cv2.putText(
                frame,
                f"GESTURE: {gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"SAMPLES: {progress}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"TIME: {elapsed:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if hand_detected:

                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO HAND",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


            cv2.putText(
                frame,
                "Q = Stop Session",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ----------------------------------------------------
        # SHOW FRAME
        # ----------------------------------------------------

        cv2.imshow(
            "Dataset V3 Collection",
            frame
        )


        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # SPACE
        if key == 32:

            if not started:

                started = True

                start_time = (
                    time.time()
                )

                print()

                print(
                    f"STARTED: {gesture}"
                )

                print(
                    f"Session ID: {session_id}"
                )

                print()

                print(
                    "Move naturally while maintaining "
                    "the gesture."
                )

            else:

                print(
                    "Already collecting..."
                )


        # Q
        elif key == ord("q"):

            print()

            print(
                "Session cancelled by user."
            )

            cap.release()

            cv2.destroyAllWindows()

            return False


        # ----------------------------------------------------
        # TARGET REACHED
        # ----------------------------------------------------

        if (
            len(samples)
            >=
            TARGET_SAMPLES
        ):

            print()

            print(
                f"SUCCESS: Target of "
                f"{TARGET_SAMPLES} samples reached."
            )

            break


    # ========================================================
    # CLOSE CAMERA
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


    # ========================================================
    # CHECK SAMPLES
    # ========================================================

    if len(samples) < 100:

        print()

        print(
            f"WARNING: Only {len(samples)} samples collected."
        )

        print(
            "This session is too small."
        )

        print(
            "It will NOT be saved."
        )

        return False


    # ========================================================
    # APPEND TO CSV
    # ========================================================

    print()

    print(
        "Saving session..."
    )


    try:

        with open(
            DATASET_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )


            for features in samples:

                writer.writerow(
                    features
                    +
                    [
                        gesture,
                        session_id
                    ]
                )


    except Exception as e:

        print()

        print(
            "ERROR: Could not save session."
        )

        print(
            e
        )

        return False


    # ========================================================
    # SESSION SUMMARY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SESSION SAVED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Gesture: {gesture}"
    )

    print(
        f"Session ID: {session_id}"
    )

    print(
        f"Samples: {len(samples)}"
    )

    print()

    print(
        f"Saved to:\n{DATASET_PATH}"
    )

    return True


# ============================================================
# 12. CREATE MEDIAPIPE LANDMARKER
# ============================================================

print()

print(
    "Loading MediaPipe Hand Landmarker..."
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            MODEL_PATH
        )
    ),

    running_mode=(
        VisionRunningMode.VIDEO
    ),

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


# ============================================================
# 13. COLLECT ALL SIX SESSIONS
# ============================================================

successful_sessions = []

cancelled = False


with HandLandmarker.create_from_options(
    options
) as landmarker:

    for gesture in GESTURES:

        print()

        print(
            "#" * 70
        )

        print(
            f"NEXT GESTURE: {gesture.upper()}"
        )

        print(
            "#" * 70
        )

        print()

        print(
            "Get ready."
        )

        print(
            "Press SPACE inside the camera window."
        )

        print()

        success = collect_session(
            gesture,
            landmarker
        )


        if success:

            successful_sessions.append(
                gesture
            )

        else:

            print()

            print(
                f"Session for '{gesture}' "
                f"was not completed."
            )

            answer = input(
                "Continue to the next gesture? (y/n): "
            ).strip().lower()

            if answer != "y":

                cancelled = True

                break


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "DATASET V3 COLLECTION SUMMARY"
)

print("=" * 70)

print()

print(
    f"Successful new sessions: "
    f"{len(successful_sessions)}"
)

print()

for gesture in GESTURES:

    if gesture in successful_sessions:

        print(
            f"  [OK] {gesture}"
        )

    else:

        print(
            f"  [--] {gesture}"
        )


print()

print(
    f"Dataset location:\n{DATASET_PATH}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "Do NOT edit the CSV manually."
)

print(
    "Do NOT delete the existing sessions."
)

print()

print(
    "After collecting all six new sessions,"
)

print(
    "run train_model_v3.py."
)

print()

print(
    "The V3 trainer will then attempt a "
    "true session-aware six-class evaluation."
)

print()

print("=" * 70)