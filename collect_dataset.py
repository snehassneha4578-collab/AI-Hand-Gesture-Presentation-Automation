import cv2
import mediapipe as mp
import csv
import os

from mediapipe.tasks.python import vision


# ============================================================
# AI HAND GESTURE DATASET COLLECTOR
# ============================================================

print("\n==========================================")
print(" AI HAND GESTURE DATASET COLLECTOR")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)

DATASET_FOLDER = os.path.join(
    BASE_DIR,
    "dataset"
)

os.makedirs(DATASET_FOLDER, exist_ok=True)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found!")
    print(MODEL_PATH)
    exit()

print("Model found:")
print(MODEL_PATH)


# ============================================================
# 3. MEDIAPIPE
# ============================================================

base_options = mp.tasks.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe initialized successfully.")


# ============================================================
# 4. GESTURES
# ============================================================

gestures = {
    "0": "fist",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "open_palm",
    "5": "pinch"
}

print("\nAvailable gestures:")
print("------------------------------------------")

for number, name in gestures.items():
    print(number, "->", name)


# ============================================================
# 5. SELECT GESTURE
# ============================================================

choice = input(
    "\nEnter the gesture number you want to collect: "
).strip()

if choice not in gestures:

    print("Invalid gesture number.")
    exit()

gesture_name = gestures[choice]

print("\nSelected gesture:", gesture_name)


# ============================================================
# 6. INSTRUCTIONS
# ============================================================

print("\n==========================================")
print(" COLLECTION INSTRUCTIONS")
print("==========================================")

print("Show the selected gesture clearly.")
print("Press SPACE once to START.")
print("The program will collect automatically.")
print("Press SPACE again to PAUSE.")
print("Press Q to QUIT.")

print("\nFor ONE:")
print("Show: INDEX FINGER UP ☝️")
print("Keep the other fingers folded.")


# ============================================================
# 7. CSV
# ============================================================

csv_path = os.path.join(
    DATASET_FOLDER,
    "gesture_dataset.csv"
)

file_exists = os.path.exists(csv_path)

file = open(
    csv_path,
    "a",
    newline=""
)

writer = csv.writer(file)


# ============================================================
# 8. HEADER
# ============================================================

if not file_exists:

    header = []

    for i in range(21):

        header.append(f"x_{i}")
        header.append(f"y_{i}")
        header.append(f"z_{i}")

    header.append("label")

    writer.writerow(header)

    file.flush()


# ============================================================
# 9. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    file.close()

    exit()


print("\nCamera opened successfully.")


# ============================================================
# 10. VARIABLES
# ============================================================

collecting = False

sample_count = 0


# ============================================================
# 11. MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read camera frame.")

        break


    # --------------------------------------------------------
    # MIRROR
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------------
    # DETECT HAND
    # --------------------------------------------------------

    result = landmarker.detect(mp_image)


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        for landmark in hand:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # DRAW CONNECTIONS
        # ----------------------------------------------------

        connections = [

            (0, 1), (1, 2), (2, 3), (3, 4),

            (5, 6), (6, 7), (7, 8),

            (9, 10), (10, 11), (11, 12),

            (13, 14), (14, 15), (15, 16),

            (17, 18), (18, 19), (19, 20),

            (0, 5), (5, 9), (9, 13),
            (13, 17), (0, 17)
        ]


        for start_index, end_index in connections:

            start = hand[start_index]
            end = hand[end_index]

            x1 = int(start.x * width)
            y1 = int(start.y * height)

            x2 = int(end.x * width)
            y2 = int(end.y * height)

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
        # AUTOMATIC DATA COLLECTION
        # ====================================================

        if collecting:

            features = []

            for landmark in hand:

                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z)

            features.append(gesture_name)

            writer.writerow(features)

            file.flush()

            sample_count += 1


    else:

        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

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
    # GESTURE
    # ========================================================

    cv2.putText(
        frame,
        f"Label: {gesture_name}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAMPLE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # COLLECTION STATUS
    # ========================================================

    if collecting:

        status = "COLLECTING"

        status_color = (0, 255, 0)

    else:

        status = "PAUSED"

        status_color = (0, 255, 255)


    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "SPACE = Start/Pause",
        (20, height - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "AI Gesture Dataset Collector",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord(" "):

        collecting = not collecting

        if collecting:

            print(
                f"\nCollecting '{gesture_name}'..."
            )

        else:

            print(
                f"\nCollection paused."
            )

            print(
                f"Samples collected: {sample_count}"
            )


    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

file.close()

cv2.destroyAllWindows()


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print(" DATA COLLECTION COMPLETE")
print("==========================================")

print("Gesture:", gesture_name)

print("Samples collected:", sample_count)

print("\nDataset saved:")
print(csv_path)