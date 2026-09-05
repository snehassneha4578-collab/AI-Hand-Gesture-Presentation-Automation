import cv2
import mediapipe as mp
import csv
import os
import time
import uuid
import math

from mediapipe.tasks.python import vision


# ============================================================
# AI HAND GESTURE DATASET COLLECTOR V2
# ============================================================
#
# Project:
# AI-Based Hand Gesture Recognition &
# Intelligent Presentation Automation System
#
# Gestures:
# 0 -> fist
# 1 -> one
# 2 -> two
# 3 -> three
# 4 -> open_palm
# 5 -> pinch
#
# Each sample contains:
# 21 landmarks x 3 coordinates
# = 63 features
#
# Plus:
# label
# session_id
#
# Total columns = 65
#
# Dataset target:
# 250 samples per gesture
#
# IMPORTANT:
# Run this program once for each gesture.
# The program safely APPENDS new sessions to the same CSV.
#
# ============================================================


print("\n")
print("=" * 70)
print(" AI HAND GESTURE DATASET COLLECTOR V2")
print("=" * 70)
print("\n")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


DATASET_FOLDER = os.path.join(
    BASE_DIR,
    "dataset"
)


# Create dataset folder if it does not exist
os.makedirs(
    DATASET_FOLDER,
    exist_ok=True
)


CSV_PATH = os.path.join(
    DATASET_FOLDER,
    "gesture_dataset_v2.csv"
)


# ============================================================
# 2. COLLECTION SETTINGS
# ============================================================

FRAME_INTERVAL = 6

MIN_MOVEMENT = 0.008

TARGET_SAMPLES = 250

MAX_SESSION_SECONDS = 90


# MediaPipe confidence settings
MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


# Expected feature count
EXPECTED_FEATURES = 63

# 63 features + label + session_id
EXPECTED_COLUMNS = 65


# ============================================================
# 3. CHECK MODEL FILE
# ============================================================

print("[1] CHECKING MODEL")
print("-" * 70)


if not os.path.exists(MODEL_PATH):

    print("ERROR: MediaPipe model was not found.")

    print("\nExpected model location:")
    print(MODEL_PATH)

    raise SystemExit


print("Model found:")
print(MODEL_PATH)


# ============================================================
# 4. CREATE MEDIAPIPE HAND LANDMARKER
# ============================================================

print("\n[2] INITIALIZING MEDIAPIPE")
print("-" * 70)


base_options = mp.tasks.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)


landmarker = vision.HandLandmarker.create_from_options(
    options
)


print("MediaPipe initialized successfully.")


# ============================================================
# 5. GESTURE MAPPING
# ============================================================

gestures = {
    "0": "fist",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "open_palm",
    "5": "pinch"
}


print("\n[3] AVAILABLE GESTURES")
print("-" * 70)


for number, gesture in gestures.items():

    print(f"{number} -> {gesture}")


# ============================================================
# 6. SELECT GESTURE
# ============================================================

choice = input(
    "\nEnter the gesture number to collect (0-5): "
).strip()


if choice not in gestures:

    print("\nERROR: Invalid gesture number.")

    landmarker.close()

    raise SystemExit


gesture_name = gestures[choice]


print("\nSelected gesture:")
print(gesture_name)


# ============================================================
# 7. CREATE EXPECTED CSV HEADER
# ============================================================

def create_header():

    header = []


    # 21 landmarks
    for i in range(21):

        header.append(f"x_{i}")
        header.append(f"y_{i}")
        header.append(f"z_{i}")


    # Label
    header.append("label")


    # Session ID
    header.append("session_id")


    return header


EXPECTED_HEADER = create_header()


# ============================================================
# 8. VALIDATE EXISTING CSV
# ============================================================

print("\n[4] CHECKING DATASET FILE")
print("-" * 70)


file_mode = "w"
write_header = True


if not os.path.exists(CSV_PATH):

    print("No existing dataset found.")

    print("A new dataset will be created.")


else:

    print("Existing dataset found.")

    print("Validating CSV header...")


    try:

        with open(
            CSV_PATH,
            "r",
            newline="",
            encoding="utf-8"
        ) as check_file:

            reader = csv.reader(check_file)

            existing_header = next(
                reader,
                None
            )


        # ----------------------------------------------------
        # HEADER CHECK
        # ----------------------------------------------------

        if existing_header != EXPECTED_HEADER:

            print("\n")
            print("=" * 70)
            print("ERROR: INVALID DATASET FILE")
            print("=" * 70)


            print(
                "\nThe existing gesture_dataset_v2.csv "
                "does not have the expected header."
            )


            print(
                "\nExpected columns:",
                EXPECTED_COLUMNS
            )


            print(
                "Found columns:",
                len(existing_header)
                if existing_header
                else 0
            )


            print(
                "\nThe existing file will NOT be modified."
            )


            print(
                "\nPlease rename or delete the invalid file:"
            )


            print(CSV_PATH)


            landmarker.close()

            raise SystemExit


        # ----------------------------------------------------
        # VALID CSV
        # ----------------------------------------------------

        file_mode = "a"

        write_header = False


        print("Dataset header: VALID")

        print(
            "New samples will be safely appended."
        )


    except UnicodeDecodeError:

        print(
            "\nERROR: Existing CSV is not a valid UTF-8 file."
        )


        landmarker.close()

        raise SystemExit


# ============================================================
# 9. CREATE UNIQUE SESSION ID
# ============================================================

session_id = (
    time.strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(uuid.uuid4())[:6]
)


print("\nSession ID:")
print(session_id)


# ============================================================
# 10. COLLECTION INSTRUCTIONS
# ============================================================

print("\n")
print("=" * 70)
print(" COLLECTION INSTRUCTIONS")
print("=" * 70)


print("\nGesture:")
print(gesture_name)


print("\nTarget:")
print(f"{TARGET_SAMPLES} samples")


print("\nIMPORTANT:")
print("Keep the correct gesture shape.")
print("Do not hold your hand in exactly one position.")


print("\nMove your hand naturally through:")

print("  LEFT")
print("  CENTER")
print("  RIGHT")

print("  TOP")
print("  MIDDLE")
print("  BOTTOM")

print("  CLOSER")
print("  FARTHER")

print("  Slight rotation")
print("  Slight tilt")

print("\n")


print("Recommended collection pattern:")
print("------------------------------------------")
print("1. Start near center")
print("2. Move slowly to the left")
print("3. Move back to center")
print("4. Move to the right")
print("5. Move upward")
print("6. Move downward")
print("7. Move slightly closer")
print("8. Move slightly farther")
print("9. Rotate/tilt slightly")
print("10. Repeat naturally")
print("------------------------------------------")


print("\nControls:")
print("------------------------------------------")
print("SPACE = Start / Pause collection")
print("Q     = Quit")
print("------------------------------------------")


# ============================================================
# 11. OPEN CSV FILE
# ============================================================

try:

    file = open(
        CSV_PATH,
        file_mode,
        newline="",
        encoding="utf-8"
    )

except Exception as error:

    print("\nERROR: Could not open dataset file.")

    print(error)

    landmarker.close()

    raise SystemExit


writer = csv.writer(file)


# ============================================================
# 12. WRITE HEADER IF REQUIRED
# ============================================================

if write_header:

    writer.writerow(
        EXPECTED_HEADER
    )

    file.flush()

    print(
        "\nNew dataset header created."
    )


# ============================================================
# 13. OPEN WEBCAM
# ============================================================

print("\n[5] OPENING WEBCAM")
print("-" * 70)


cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )


    file.close()

    landmarker.close()

    raise SystemExit


print("Camera opened successfully.")


# ============================================================
# 14. VARIABLES
# ============================================================

collecting = False

sample_count = 0

frame_counter = 0

start_time = None

last_features = None


# ============================================================
# 15. MOVEMENT CALCULATION
# ============================================================

def calculate_movement(
    current_features,
    previous_features
):

    # No previous sample
    if previous_features is None:

        return 999.0


    total_distance = 0.0


    # 21 landmarks
    for i in range(21):

        # Current landmark
        x1 = current_features[i * 3]
        y1 = current_features[i * 3 + 1]
        z1 = current_features[i * 3 + 2]


        # Previous landmark
        x2 = previous_features[i * 3]
        y2 = previous_features[i * 3 + 1]
        z2 = previous_features[i * 3 + 2]


        # Euclidean distance
        distance = math.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
            +
            (z1 - z2) ** 2
        )


        total_distance += distance


    # Average movement across 21 landmarks
    average_movement = (
        total_distance / 21
    )


    return average_movement


# ============================================================
# 16. FEATURE VALIDATION
# ============================================================

def valid_features(features):

    # Must contain exactly 63 values
    if len(features) != EXPECTED_FEATURES:

        return False


    for value in features:

        # Must be numeric
        if not isinstance(
            value,
            (int, float)
        ):

            return False


        # Must be finite
        if not math.isfinite(value):

            return False


        # Reject impossible extreme values
        if abs(value) > 5:

            return False


    return True


# ============================================================
# 17. DRAW HAND CONNECTIONS
# ============================================================

connections = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (13, 14),
    (14, 15),
    (15, 16),

    # Little finger
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17)
]


# ============================================================
# 18. MAIN CAMERA LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()


    if not success:

        print(
            "\nERROR: Could not read webcam frame."
        )

        break


    # --------------------------------------------------------
    # MIRROR IMAGE
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # FRAME DIMENSIONS
    # --------------------------------------------------------

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # CONVERT BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------------
    # DETECT HAND
    # --------------------------------------------------------

    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # ----------------------------------------------------
        # DRAW LANDMARK POINTS
        # ----------------------------------------------------

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )


            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # DRAW CONNECTIONS
        # ----------------------------------------------------

        for start_index, end_index in connections:

            start = hand[start_index]

            end = hand[end_index]


            x1 = int(
                start.x * width
            )

            y1 = int(
                start.y * height
            )


            x2 = int(
                end.x * width
            )

            y2 = int(
                end.y * height
            )


            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # HAND STATUS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ====================================================
        # EXTRACT 63 FEATURES
        # ====================================================

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


        # ====================================================
        # VALIDATE FEATURES
        # ====================================================

        features_valid = valid_features(
            features
        )


        # ====================================================
        # DATA COLLECTION
        # ====================================================

        if collecting and features_valid:

            frame_counter += 1


            # ------------------------------------------------
            # FRAME INTERVAL
            # ------------------------------------------------

            if (
                frame_counter
                % FRAME_INTERVAL
                == 0
            ):

                movement = calculate_movement(
                    features,
                    last_features
                )


                should_save = False


                # ------------------------------------------------
                # FIRST SAMPLE
                # ------------------------------------------------

                if last_features is None:

                    should_save = True


                # ------------------------------------------------
                # MOVEMENT FILTER
                # ------------------------------------------------

                elif movement >= MIN_MOVEMENT:

                    should_save = True


                # ------------------------------------------------
                # SAVE SAMPLE
                # ------------------------------------------------

                if should_save:

                    row = list(
                        features
                    )


                    # Add gesture label
                    row.append(
                        gesture_name
                    )


                    # Add session ID
                    row.append(
                        session_id
                    )


                    # Write row
                    writer.writerow(
                        row
                    )


                    # Immediately save to disk
                    file.flush()


                    # Update counter
                    sample_count += 1


                    # Update previous features
                    last_features = list(
                        features
                    )


                    # Console output
                    print(
                        f"Saved sample "
                        f"{sample_count}/{TARGET_SAMPLES}"
                        f" | movement={movement:.4f}"
                    )


        elif collecting and not features_valid:

            cv2.putText(
                frame,
                "INVALID LANDMARK DATA",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY GESTURE
    # ========================================================

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY SAMPLE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{TARGET_SAMPLES}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY SESSION ID
    # ========================================================

    cv2.putText(
        frame,
        f"Session: {session_id[-6:]}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # TIMER
    # ========================================================

    if collecting and start_time is not None:

        elapsed = (
            time.time()
            - start_time
        )


        remaining = max(
            0,
            MAX_SESSION_SECONDS - elapsed
        )


        cv2.putText(
            frame,
            f"Time: {remaining:.0f}s",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # TIME LIMIT
        # ----------------------------------------------------

        if elapsed >= MAX_SESSION_SECONDS:

            collecting = False


            print("\n")
            print("=" * 70)
            print("TIME LIMIT REACHED")
            print("=" * 70)


            print(
                "Gesture:",
                gesture_name
            )


            print(
                "Samples collected:",
                sample_count
            )


            print(
                "Session ID:",
                session_id
            )


            print(
                "\nPress Q to close the camera."
            )


    # ========================================================
    # AUTOMATIC TARGET STOP
    # ========================================================

    if sample_count >= TARGET_SAMPLES:

        collecting = False


        print("\n")
        print("=" * 70)
        print("TARGET REACHED")
        print("=" * 70)


        print(
            "Gesture:",
            gesture_name
        )


        print(
            "Samples collected:",
            sample_count
        )


        print(
            "Session ID:",
            session_id
        )


        print(
            "\nPress Q to close the camera."
        )


    # ========================================================
    # STATUS DISPLAY
    # ========================================================

    if collecting:

        status_text = "COLLECTING"

        status_color = (0, 255, 0)

    else:

        status_text = "PAUSED"

        status_color = (0, 255, 255)


    cv2.putText(
        frame,
        f"Status: {status_text}",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )


    # ========================================================
    # INSTRUCTIONS ON SCREEN
    # ========================================================

    cv2.putText(
        frame,
        "SPACE: Start/Pause",
        (20, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q: Quit",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA WINDOW
    # ========================================================

    cv2.imshow(
        "AI Hand Gesture Dataset Collector V2",
        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # SPACE -> START / PAUSE
    # --------------------------------------------------------

    if key == ord(" "):

        if not collecting:

            # Do not restart after target reached
            if sample_count < TARGET_SAMPLES:

                collecting = True

                start_time = time.time()

                frame_counter = 0

                last_features = None


                print("\n")
                print("=" * 70)
                print("COLLECTION STARTED")
                print("=" * 70)


                print(
                    "Gesture:",
                    gesture_name
                )


                print(
                    "Session:",
                    session_id
                )


        else:

            collecting = False


            print("\nCollection paused.")


    # --------------------------------------------------------
    # Q -> QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        print("\n")
        print("=" * 70)
        print("STOPPING COLLECTION")
        print("=" * 70)


        print(
            "Gesture:",
            gesture_name
        )


        print(
            "Samples collected:",
            sample_count
        )


        print(
            "Session ID:",
            session_id
        )


        break


# ============================================================
# 19. CLEANUP
# ============================================================

print("\nCleaning up...")


cap.release()

cv2.destroyAllWindows()

file.close()

landmarker.close()


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("DATA COLLECTION FINISHED")
print("=" * 70)


print(
    "Gesture:",
    gesture_name
)


print(
    "Samples collected:",
    sample_count
)


print(
    "Session ID:",
    session_id
)


print(
    "\nDataset file:"
)


print(CSV_PATH)


print("\n")


if sample_count >= TARGET_SAMPLES:

    print(
        "SUCCESS: Target of 250 samples reached."
    )

else:

    print(
        "NOTE: Target was not completely reached."
    )


print("\nNext time, run the same file and select")
print("a different gesture number.")


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)