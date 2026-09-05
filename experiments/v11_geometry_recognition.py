import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")
REPORT = os.path.join(BASE_DIR, "dataset", "v11_geometry_results.txt")

GESTURES = ["fist", "one", "open_palm", "pinch", "three", "two"]

# MediaPipe landmark indices
WRIST = 0

THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def point(row, index):
    return np.array([
        float(row[f"x_{index}"]),
        float(row[f"y_{index}"]),
        float(row[f"z_{index}"])
    ])


def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):
    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom < 1e-9:
        return 0.0

    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def finger_extension(row, mcp, pip, dip, tip):

    wrist = point(row, WRIST)

    p_mcp = point(row, mcp)
    p_pip = point(row, pip)
    p_dip = point(row, dip)
    p_tip = point(row, tip)

    wrist_tip = distance(wrist, p_tip)
    wrist_mcp = distance(wrist, p_mcp)

    ratio = wrist_tip / (wrist_mcp + 1e-6)

    pip_angle = angle(p_mcp, p_pip, p_dip)
    dip_angle = angle(p_pip, p_dip, p_tip)

    extended = (
        ratio > 1.35
        and pip_angle > 145
        and dip_angle > 140
    )

    return extended


def thumb_state(row):

    wrist = point(row, WRIST)
    mcp = point(row, THUMB_MCP)
    ip = point(row, THUMB_IP)
    tip = point(row, THUMB_TIP)

    wrist_tip = distance(wrist, tip)
    wrist_mcp = distance(wrist, mcp)

    ratio = wrist_tip / (wrist_mcp + 1e-6)

    ip_angle = angle(mcp, ip, tip)

    return (
        ratio > 1.25
        and ip_angle > 145
    )


def get_finger_states(row):

    thumb = thumb_state(row)

    index = finger_extension(
        row,
        INDEX_MCP,
        INDEX_PIP,
        INDEX_DIP,
        INDEX_TIP
    )

    middle = finger_extension(
        row,
        MIDDLE_MCP,
        MIDDLE_PIP,
        MIDDLE_DIP,
        MIDDLE_TIP
    )

    ring = finger_extension(
        row,
        RING_MCP,
        RING_PIP,
        RING_DIP,
        RING_TIP
    )

    pinky = finger_extension(
        row,
        PINKY_MCP,
        PINKY_PIP,
        PINKY_DIP,
        PINKY_TIP
    )

    return [thumb, index, middle, ring, pinky]


def classify_geometry(row):

    thumb, index, middle, ring, pinky = get_finger_states(row)

    # -----------------------------------------------------
    # PINCH
    # -----------------------------------------------------

    thumb_tip = point(row, THUMB_TIP)
    index_tip = point(row, INDEX_TIP)

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    wrist = point(row, WRIST)
    middle_mcp = point(row, MIDDLE_MCP)

    palm_scale = distance(
        wrist,
        middle_mcp
    ) + 1e-6

    normalized_pinch = (
        pinch_distance / palm_scale
    )

    if normalized_pinch < 0.55:
        return "pinch"

    # -----------------------------------------------------
    # FINGER COUNT
    # -----------------------------------------------------

    finger_count = sum([
        index,
        middle,
        ring,
        pinky
    ])

    total_extended = sum([
        thumb,
        index,
        middle,
        ring,
        pinky
    ])

    # Open palm
    if total_extended >= 4:
        return "open_palm"

    # Fist
    if total_extended == 0:
        return "fist"

    # One / two / three
    if finger_count == 1:
        return "one"

    if finger_count == 2:
        return "two"

    if finger_count == 3:
        return "three"

    # Fallback
    if total_extended <= 1:
        return "fist"

    if total_extended >= 4:
        return "open_palm"

    if finger_count == 1:
        return "one"

    if finger_count == 2:
        return "two"

    return "three"


print("=" * 75)
print("V11 GEOMETRIC FINGER-STATE RECOGNITION")
print("=" * 75)

df = pd.read_csv(DATASET)

print(f"\nDataset rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Gestures: {df['label'].nunique()}")
print(f"Sessions: {df['session_id'].nunique()}")

# Verify expected landmark columns
required_columns = []

for i in range(21):
    required_columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing landmark columns: {missing}"
    )

print("\nLandmark structure verified:")
print("21 landmarks x 3 coordinates = 63 features")

# ---------------------------------------------------------
# Session-independent split
# ---------------------------------------------------------

train_parts = []
test_parts = []

print("\nSESSION-INDEPENDENT SPLIT")

for gesture in GESTURES:

    sessions = sorted(
        df[
            df["label"] == gesture
        ]["session_id"].unique()
    )

    if len(sessions) < 3:
        raise ValueError(
            f"{gesture} has fewer than 3 sessions."
        )

    train_sessions = sessions[:2]
    test_session = sessions[2]

    print(f"\n{gesture}")
    print(f"  TRAIN: {train_sessions}")
    print(f"  TEST : {test_session}")

    train_parts.append(
        df[
            (df["label"] == gesture)
            &
            (df["session_id"].isin(train_sessions))
        ]
    )

    test_parts.append(
        df[
            (df["label"] == gesture)
            &
            (df["session_id"] == test_session)
        ]
    )

train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)

print("\nTraining/reference samples:", len(train_df))
print("Unseen test samples:       ", len(test_df))

# ---------------------------------------------------------
# GEOMETRIC RECOGNITION
# ---------------------------------------------------------

print("\nRunning geometric recognizer...")

y_true = test_df["label"].values

predictions = []

for _, row in test_df.iterrows():

    predictions.append(
        classify_geometry(row)
    )

y_pred = np.array(predictions)

accuracy = accuracy_score(
    y_true,
    y_pred
)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("V11 RESULTS")
print("=" * 75)

print(
    f"\nOverall unseen-session accuracy: "
    f"{accuracy * 100:.2f}%"
)

report = classification_report(
    y_true,
    y_pred,
    labels=GESTURES,
    target_names=GESTURES,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=GESTURES
)

print("CONFUSION MATRIX")

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

print(cm_df.to_string())

print("\nPER-GESTURE ACCURACY")

per_gesture = {}

for i, gesture in enumerate(GESTURES):

    total = cm[i].sum()
    correct = cm[i, i]

    acc = (
        correct / total * 100
        if total > 0
        else 0
    )

    per_gesture[gesture] = acc

    print(
        f"{gesture:10s}: "
        f"{acc:6.2f}% "
        f"({correct}/{total})"
    )

# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V11 GEOMETRIC FINGER-STATE RECOGNITION\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write(
        f"Dataset rows: {len(df)}\n"
    )

    f.write(
        f"Columns: {len(df.columns)}\n"
    )

    f.write(
        f"Gestures: {df['label'].nunique()}\n"
    )

    f.write(
        f"Sessions: {df['session_id'].nunique()}\n\n"
    )

    f.write(
        f"Training/reference samples: "
        f"{len(train_df)}\n"
    )

    f.write(
        f"Unseen test samples: "
        f"{len(test_df)}\n\n"
    )

    f.write(
        f"Overall unseen-session accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(report)

    f.write(
        "\nCONFUSION MATRIX\n"
    )

    f.write(
        cm_df.to_string()
    )

    f.write(
        "\n\nPER-GESTURE ACCURACY\n"
    )

    for gesture, acc in per_gesture.items():

        f.write(
            f"{gesture:10s}: "
            f"{acc:.2f}%\n"
        )

print("\nReport saved to:")
print(REPORT)

print("\n" + "=" * 75)
print("V11 ANALYSIS COMPLETE")
print("=" * 75)
