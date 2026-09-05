from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "gesture_dataset_v2.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "trained_model"
)

MODEL_PATH = (
    MODEL_DIR
    / "gesture_svm_v3.pkl"
)

SCALER_PATH = (
    MODEL_DIR
    / "scaler_v3.pkl"
)

FEATURE_DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "gesture_dataset_v3_features.csv"
)

RESULTS_PATH = (
    BASE_DIR
    / "dataset"
    / "v3_evaluation_results.txt"
)

RANDOM_STATE = 42

SVM_C = 10

SVM_GAMMA = "scale"

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]

LANDMARK_COUNT = 21


# ============================================================
# RAW FEATURE COLUMNS
# ============================================================

RAW_FEATURE_COLUMNS = []

for i in range(LANDMARK_COUNT):

    RAW_FEATURE_COLUMNS.append(
        f"x_{i}"
    )

    RAW_FEATURE_COLUMNS.append(
        f"y_{i}"
    )

    RAW_FEATURE_COLUMNS.append(
        f"z_{i}"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("AI HAND GESTURE RECOGNITION - V3 TRAINING")
print("=" * 70)

print()
print("V3 = Shape-Based Hand Geometry")
print()
print("Evaluation = Independent Session Holdout")
print()

print(
    f"Dataset:\n{DATASET_PATH}"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_PATH.exists():

    print()
    print("ERROR: Dataset not found.")

    print(
        f"Expected:\n{DATASET_PATH}"
    )

    sys.exit(1)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

try:

    df = pd.read_csv(
        DATASET_PATH
    )

except Exception as e:

    print()
    print(
        "ERROR: Could not read dataset."
    )

    print(
        f"Reason: {e}"
    )

    sys.exit(1)


print()
print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

print()
print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

required_columns = (
    RAW_FEATURE_COLUMNS
    +
    [
        "label",
        "session_id"
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print()
    print(
        "ERROR: Missing columns:"
    )

    for column in missing_columns:

        print(
            f"  - {column}"
        )

    sys.exit(1)


print()
print(
    "63 raw landmark features found."
)

print(
    "label column found."
)

print(
    "session_id column found."
)


# ============================================================
# GESTURE DISTRIBUTION
# ============================================================

print()
print("Gesture distribution:")

label_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

for gesture, count in label_counts.items():

    print(
        f"  {gesture:<12} {count}"
    )


# ============================================================
# UNKNOWN LABEL CHECK
# ============================================================

unknown_labels = (
    set(df["label"].unique())
    -
    set(GESTURES)
)

if unknown_labels:

    print()
    print(
        "ERROR: Unknown gesture labels:"
    )

    for gesture in unknown_labels:

        print(
            f"  - {gesture}"
        )

    sys.exit(1)


# ============================================================
# SESSION SUMMARY
# ============================================================

print()
print(
    "Session information:"
)

total_sessions = (
    df["session_id"]
    .nunique()
)

print(
    f"Total sessions: {total_sessions}"
)


session_summary = (
    df.groupby(
        [
            "session_id",
            "label"
        ]
    )
    .size()
    .reset_index(
        name="samples"
    )
)

print()

for _, row in session_summary.iterrows():

    print(
        f"  {row['session_id']} -> "
        f"{row['label']} -> "
        f"{row['samples']} samples"
    )


# ============================================================
# SESSIONS PER GESTURE
# ============================================================

print()
print(
    "Sessions per gesture:"
)

sessions_per_gesture = (
    df.groupby(
        "label"
    )["session_id"]
    .nunique()
)

for gesture in GESTURES:

    count = int(
        sessions_per_gesture.get(
            gesture,
            0
        )
    )

    print(
        f"  {gesture:<12} "
        f"{count} session(s)"
    )


# ============================================================
# REQUIRE AT LEAST TWO SESSIONS
# ============================================================

insufficient_gestures = []

for gesture in GESTURES:

    count = int(
        sessions_per_gesture.get(
            gesture,
            0
        )
    )

    if count < 2:

        insufficient_gestures.append(
            gesture
        )


if insufficient_gestures:

    print()
    print(
        "ERROR: The following gestures"
    )

    print(
        "do not have at least two sessions:"
    )

    for gesture in insufficient_gestures:

        print(
            f"  - {gesture}"
        )

    print()
    print(
        "Collect another independent session"
    )

    print(
        "for those gestures."
    )

    sys.exit(1)


print()
print(
    "SUCCESS: Every gesture has at least"
)

print(
    "two independent sessions."
)


# ============================================================
# MISSING VALUES
# ============================================================

print()
print(
    "Checking missing values..."
)

missing_count = (
    df[
        RAW_FEATURE_COLUMNS
    ]
    .isna()
    .sum()
    .sum()
)

print(
    f"Missing feature values: "
    f"{missing_count}"
)

if missing_count > 0:

    print()
    print(
        "ERROR: Missing values detected."
    )

    sys.exit(1)


# ============================================================
# FINITE VALUES
# ============================================================

print()
print(
    "Checking finite values..."
)

raw_array = (
    df[
        RAW_FEATURE_COLUMNS
    ]
    .to_numpy(
        dtype=np.float64
    )
)

non_finite_count = (
    ~np.isfinite(
        raw_array
    )
).sum()

print(
    f"Non-finite values: "
    f"{non_finite_count}"
)

if non_finite_count > 0:

    print()
    print(
        "ERROR: Non-finite values detected."
    )

    sys.exit(1)


# ============================================================
# ANGLE FUNCTION
# ============================================================

def calculate_angle(
    a,
    b,
    c
):

    ba = a - b

    bc = c - b

    norm_ba = np.linalg.norm(
        ba
    )

    norm_bc = np.linalg.norm(
        bc
    )

    if (
        norm_ba < 1e-8
        or
        norm_bc < 1e-8
    ):

        return 0.0

    cosine_angle = (
        np.dot(
            ba,
            bc
        )
        /
        (
            norm_ba
            *
            norm_bc
        )
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(
            cosine_angle
        )
    )

    return float(
        angle
    )


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(
    a,
    b
):

    return float(
        np.linalg.norm(
            a - b
        )
    )


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(
    landmarks
):

    # Wrist
    wrist = landmarks[0]

    # Move wrist to origin
    relative = (
        landmarks
        -
        wrist
    )

    # Wrist -> middle MCP
    scale = np.linalg.norm(
        relative[9]
    )

    # Fallback
    if scale < 1e-6:

        mcp_indices = [
            5,
            9,
            13,
            17
        ]

        distances = []

        for index in mcp_indices:

            distances.append(
                np.linalg.norm(
                    relative[index]
                )
            )

        if distances:

            scale = np.mean(
                distances
            )

    # Final fallback
    if scale < 1e-6:

        scale = 1.0

    normalized = (
        relative
        /
        scale
    )

    return normalized


# ============================================================
# V3 FEATURE EXTRACTION
# ============================================================

def extract_shape_features(
    raw_values
):

    raw_values = np.asarray(
        raw_values,
        dtype=np.float64
    )

    if raw_values.size != 63:

        raise ValueError(
            f"Expected 63 values, "
            f"got {raw_values.size}"
        )

    # 21 × 3
    landmarks = (
        raw_values
        .reshape(
            21,
            3
        )
    )

    # Normalize
    normalized = (
        normalize_landmarks(
            landmarks
        )
    )

    features = []

    # ========================================================
    # A. NORMALIZED LANDMARK COORDINATES
    # 63
    # ========================================================

    for landmark in normalized:

        features.extend(
            landmark.tolist()
        )


    # ========================================================
    # B. WRIST-TO-LANDMARK DISTANCES
    # 20
    # ========================================================

    wrist = normalized[0]

    for i in range(
        1,
        21
    ):

        distance = (
            calculate_distance(
                wrist,
                normalized[i]
            )
        )

        features.append(
            distance
        )


    # ========================================================
    # C. IMPORTANT HAND DISTANCES
    # 17
    # ========================================================

    important_pairs = [

        (4, 8),
        (4, 12),
        (4, 16),
        (4, 20),

        (8, 12),
        (8, 16),
        (8, 20),

        (12, 16),
        (12, 20),

        (16, 20),

        (5, 9),
        (9, 13),
        (13, 17),

        (5, 8),
        (9, 12),
        (13, 16),
        (17, 20)
    ]

    for a, b in important_pairs:

        distance = (
            calculate_distance(
                normalized[a],
                normalized[b]
            )
        )

        features.append(
            distance
        )


    # ========================================================
    # D. FINGER BONE LENGTHS
    # 16
    # ========================================================

    bone_pairs = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (5, 6),
        (6, 7),
        (7, 8),

        (9, 10),
        (10, 11),
        (11, 12),

        (13, 14),
        (14, 15),
        (15, 16),

        (17, 18),
        (18, 19),
        (19, 20)
    ]

    for a, b in bone_pairs:

        length = (
            calculate_distance(
                normalized[a],
                normalized[b]
            )
        )

        features.append(
            length
        )


    # ========================================================
    # E. JOINT ANGLES
    # 10
    # ========================================================

    angle_triplets = [

        (1, 2, 3),
        (2, 3, 4),

        (5, 6, 7),
        (6, 7, 8),

        (9, 10, 11),
        (10, 11, 12),

        (13, 14, 15),
        (14, 15, 16),

        (17, 18, 19),
        (18, 19, 20)
    ]

    for a, b, c in angle_triplets:

        angle = calculate_angle(
            normalized[a],
            normalized[b],
            normalized[c]
        )

        features.append(
            angle
        )


    # ========================================================
    # F. FINGER EXTENSION RATIOS
    # 5
    # ========================================================

    finger_pairs = [

        (4, 1),
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17)
    ]

    for tip, mcp in finger_pairs:

        tip_distance = (
            calculate_distance(
                wrist,
                normalized[tip]
            )
        )

        mcp_distance = (
            calculate_distance(
                wrist,
                normalized[mcp]
            )
        )

        ratio = (
            tip_distance
            /
            (
                mcp_distance
                +
                1e-8
            )
        )

        features.append(
            ratio
        )


    # ========================================================
    # G. FINGERTIP Y DIFFERENCES
    # 10
    # ========================================================

    fingertip_indices = [
        4,
        8,
        12,
        16,
        20
    ]

    for i in range(
        len(fingertip_indices)
    ):

        for j in range(
            i + 1,
            len(fingertip_indices)
        ):

            a = fingertip_indices[i]

            b = fingertip_indices[j]

            difference = (
                normalized[a][1]
                -
                normalized[b][1]
            )

            features.append(
                float(
                    difference
                )
            )


    # ========================================================
    # H. FINGERTIP X DIFFERENCES
    # 10
    # ========================================================

    for i in range(
        len(fingertip_indices)
    ):

        for j in range(
            i + 1,
            len(fingertip_indices)
        ):

            a = fingertip_indices[i]

            b = fingertip_indices[j]

            difference = (
                normalized[a][0]
                -
                normalized[b][0]
            )

            features.append(
                float(
                    difference
                )
            )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    features = np.asarray(
        features,
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(
            features
        )
    ):

        raise ValueError(
            "Non-finite V3 feature detected."
        )

    return features


# ============================================================
# EXTRACT ALL V3 FEATURES
# ============================================================

print()
print("=" * 70)
print("EXTRACTING V3 SHAPE FEATURES")
print("=" * 70)

print()
print(
    "Processing 21 MediaPipe landmarks..."
)

print(
    "Please wait..."
)

v3_feature_rows = []

valid_indices = []

for index, row in df.iterrows():

    try:

        raw_values = (
            row[
                RAW_FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        features = (
            extract_shape_features(
                raw_values
            )
        )

        v3_feature_rows.append(
            features
        )

        valid_indices.append(
            index
        )

    except Exception as e:

        print(
            f"WARNING: Row {index} skipped: {e}"
        )


if not v3_feature_rows:

    print()
    print(
        "ERROR: No valid V3 features generated."
    )

    sys.exit(1)


X_v3 = np.vstack(
    v3_feature_rows
)

print()
print(
    f"Original samples: "
    f"{len(df)}"
)

print(
    f"Valid samples: "
    f"{len(X_v3)}"
)

print(
    f"V3 feature count: "
    f"{X_v3.shape[1]}"
)


if X_v3.shape[1] != 151:

    print()
    print(
        "ERROR: Expected 151 V3 features."
    )

    print(
        f"Found: {X_v3.shape[1]}"
    )

    sys.exit(1)


print()
print(
    "SUCCESS: 151 V3 features generated."
)


# ============================================================
# BUILD V3 DATAFRAME
# ============================================================

valid_df = (
    df.iloc[
        valid_indices
    ]
    .reset_index(
        drop=True
    )
)

v3_feature_names = [

    f"v3_feature_{i}"

    for i in range(
        151
    )
]

v3_df = pd.DataFrame(
    X_v3,
    columns=v3_feature_names
)

v3_df["label"] = (
    valid_df[
        "label"
    ]
    .to_numpy()
)

v3_df["session_id"] = (
    valid_df[
        "session_id"
    ]
    .to_numpy()
)


# ============================================================
# SAVE V3 FEATURES
# ============================================================

print()
print(
    "Saving V3 feature dataset..."
)

v3_df.to_csv(
    FEATURE_DATASET_PATH,
    index=False
)

print()
print(
    f"Saved:\n{FEATURE_DATASET_PATH}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X = (
    v3_df[
        v3_feature_names
    ]
    .to_numpy(
        dtype=np.float64
    )
)

y = (
    v3_df[
        "label"
    ]
    .to_numpy()
)

groups = (
    v3_df[
        "session_id"
    ]
    .to_numpy()
)


# ============================================================
# SESSION-AWARE SPLIT
# ============================================================

print()
print("=" * 70)
print("SESSION-AWARE TRAIN / TEST SPLIT")
print("=" * 70)

print()
print(
    "One complete independent session"
)

print(
    "from EACH gesture will be held out"
)

print(
    "for testing."
)

print()


# ============================================================
# SELECT ONE TEST SESSION PER GESTURE
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

test_sessions = []

print(
    "Selected test sessions:"
)

for gesture in GESTURES:

    gesture_sessions = (
        df.loc[
            df["label"] == gesture,
            "session_id"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    gesture_sessions = np.sort(
        gesture_sessions
    )

    selected_session = rng.choice(
        gesture_sessions
    )

    test_sessions.append(
        selected_session
    )

    print(
        f"  {gesture:<12} "
        f"{selected_session}"
    )


test_sessions = set(
    test_sessions
)


# ============================================================
# CREATE TRAIN / TEST INDICES
# ============================================================

test_mask = np.isin(
    groups,
    list(test_sessions)
)

train_mask = ~test_mask


train_indices = np.where(
    train_mask
)[0]

test_indices = np.where(
    test_mask
)[0]


X_train = X[
    train_indices
]

X_test = X[
    test_indices
]

y_train = y[
    train_indices
]

y_test = y[
    test_indices
]

groups_train = groups[
    train_indices
]

groups_test = groups[
    test_indices
]


# ============================================================
# VERIFY CLASS COVERAGE
# ============================================================

train_classes = set(
    y_train
)

test_classes = set(
    y_test
)


if (
    train_classes != set(GESTURES)
    or
    test_classes != set(GESTURES)
):

    print()
    print(
        "ERROR: Invalid session-aware split."
    )

    print(
        "Every gesture must exist in both"
    )

    print(
        "training and testing sets."
    )

    sys.exit(1)


# ============================================================
# SPLIT SUMMARY
# ============================================================

print()

print(
    "SUCCESS: Valid six-class session split."
)

print()

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)

print(
    f"Training sessions: "
    f"{len(np.unique(groups_train))}"
)

print(
    f"Testing sessions: "
    f"{len(np.unique(groups_test))}"
)


print()
print(
    "Training distribution:"
)

for gesture in GESTURES:

    count = np.sum(
        y_train == gesture
    )

    print(
        f"  {gesture:<12} {count}"
    )


print()
print(
    "Testing distribution:"
)

for gesture in GESTURES:

    count = np.sum(
        y_test == gesture
    )

    print(
        f"  {gesture:<12} {count}"
    )


# ============================================================
# SCALE FEATURES
# ============================================================

print()
print("=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)

print()
print(
    "StandardScaler fitted on training data."
)

print(
    "Test data transformed using training scaler."
)


# ============================================================
# TRAIN SVM
# ============================================================

print()
print("=" * 70)
print("TRAINING V3 SVM")
print("=" * 70)

print()
print(
    "Algorithm: Support Vector Machine"
)

print(
    "Kernel: RBF"
)

print(
    f"C: {SVM_C}"
)

print(
    f"Gamma: {SVM_GAMMA}"
)

print(
    "Probability: enabled"
)

print()
print(
    "Training..."
)


model = SVC(
    kernel="rbf",
    C=SVM_C,
    gamma=SVM_GAMMA,
    probability=True,
    random_state=RANDOM_STATE
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore"
    )

    model.fit(
        X_train_scaled,
        y_train
    )


print()
print(
    "Training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("=" * 70)
print("V3 EVALUATION")
print("=" * 70)

y_pred = (
    model.predict(
        X_test_scaled
    )
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print(
    f"V3 Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=GESTURES,
    zero_division=0
)

print()
print(
    "Classification Report:"
)

print()
print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=GESTURES
)

print(
    "Confusion Matrix:"
)

print()

print(
    f"{'Actual / Predicted':<18}",
    end=""
)

for gesture in GESTURES:

    print(
        f"{gesture[:10]:>12}",
        end=""
    )

print()

for i, gesture in enumerate(
    GESTURES
):

    print(
        f"{gesture:<18}",
        end=""
    )

    for value in cm[i]:

        print(
            f"{value:>12}",
            end=""
        )

    print()


# ============================================================
# PER-GESTURE ACCURACY
# ============================================================

print()

print(
    "Per-Gesture Accuracy:"
)

per_gesture_results = {}

for i, gesture in enumerate(
    GESTURES
):

    total = cm[i].sum()

    correct = cm[i][i]

    if total > 0:

        gesture_accuracy = (
            correct
            /
            total
        ) * 100

    else:

        gesture_accuracy = 0.0

    per_gesture_results[
        gesture
    ] = gesture_accuracy

    print(
        f"  {gesture:<12} "
        f"{correct}/{total} "
        f"= {gesture_accuracy:.2f}%"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()

print(
    "Prediction Distribution:"
)

prediction_counts = (
    pd.Series(
        y_pred
    )
    .value_counts()
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
        /
        len(y_pred)
    ) * 100

    print(
        f"  {gesture:<12} "
        f"{count:<5} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print()

print(
    "Confidence Analysis:"
)

try:

    probabilities = (
        model.predict_proba(
            X_test_scaled
        )
    )

    max_probabilities = (
        probabilities.max(
            axis=1
        )
    )

    average_confidence = (
        max_probabilities.mean()
        *
        100
    )

    minimum_confidence = (
        max_probabilities.min()
        *
        100
    )

    maximum_confidence = (
        max_probabilities.max()
        *
        100
    )

    print(
        f"  Average confidence: "
        f"{average_confidence:.2f}%"
    )

    print(
        f"  Minimum confidence: "
        f"{minimum_confidence:.2f}%"
    )

    print(
        f"  Maximum confidence: "
        f"{maximum_confidence:.2f}%"
    )

except Exception as e:

    print(
        "Could not calculate confidence."
    )

    print(e)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("=" * 70)
print("SAVING V3 MODEL")
print("=" * 70)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print()

print(
    f"Model saved:\n{MODEL_PATH}"
)

print()

print(
    f"Scaler saved:\n{SCALER_PATH}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

print()

print(
    "Saving evaluation report..."
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI-Based Hand Gesture Recognition "
        "& Intelligent Presentation Automation\n"
    )

    file.write(
        "V3 Shape-Based SVM Evaluation\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        "DATASET\n"
    )

    file.write(
        f"Dataset: {DATASET_PATH}\n"
    )

    file.write(
        f"Total samples: {len(v3_df)}\n"
    )

    file.write(
        f"V3 features: {X.shape[1]}\n"
    )

    file.write(
        f"Total sessions: "
        f"{len(np.unique(groups))}\n"
    )

    file.write(
        "\n"
    )

    file.write(
        "MODEL\n"
    )

    file.write(
        "Algorithm: SVM\n"
    )

    file.write(
        "Kernel: RBF\n"
    )

    file.write(
        f"C: {SVM_C}\n"
    )

    file.write(
        f"Gamma: {SVM_GAMMA}\n"
    )

    file.write(
        "\n"
    )

    file.write(
        "SESSION-AWARE SPLIT\n"
    )

    file.write(
        "One complete session per gesture "
        "was held out for testing.\n"
    )

    file.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    file.write(
        f"Testing samples: "
        f"{len(X_test)}\n"
    )

    file.write(
        f"Training sessions: "
        f"{len(np.unique(groups_train))}\n"
    )

    file.write(
        f"Testing sessions: "
        f"{len(np.unique(groups_test))}\n"
    )

    file.write(
        "\n"
    )

    file.write(
        "HELD-OUT TEST SESSIONS\n"
    )

    for session in sorted(
        test_sessions
    ):

        session_gesture = (
            df.loc[
                df["session_id"] == session,
                "label"
            ]
            .iloc[0]
        )

        file.write(
            f"{session_gesture}: "
            f"{session}\n"
        )

    file.write(
        "\n"
    )

    file.write(
        "ACCURACY\n"
    )

    file.write(
        f"V3 Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        "\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        report
    )

    file.write(
        "\n"
    )

    file.write(
        "CONFUSION MATRIX\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\n"
    )

    file.write(
        "PER-GESTURE ACCURACY\n"
    )

    for gesture in GESTURES:

        file.write(
            f"{gesture}: "
            f"{per_gesture_results[gesture]:.2f}%\n"
        )

    file.write(
        "\n"
    )

    file.write(
        "PREDICTION DISTRIBUTION\n"
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
            /
            len(y_pred)
        ) * 100

        file.write(
            f"{gesture}: "
            f"{count} "
            f"({percentage:.2f}%)\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("V3 TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"V3 Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    "Session-aware test:"
)

print(
    "1 independent session per gesture"
)

print()

print(
    "Files created:"
)

print(
    f"  1. {MODEL_PATH}"
)

print(
    f"  2. {SCALER_PATH}"
)

print(
    f"  3. {FEATURE_DATASET_PATH}"
)

print(
    f"  4. {RESULTS_PATH}"
)

print()

print(
    "V3 pipeline:"
)

print(
    "21 MediaPipe landmarks"
)

print(
    "        ↓"
)

print(
    "Wrist-relative coordinates"
)

print(
    "        ↓"
)

print(
    "Scale normalization"
)

print(
    "        ↓"
)

print(
    "Distances + bone geometry"
)

print(
    "        ↓"
)

print(
    "Joint angles"
)

print(
    "        ↓"
)

print(
    "Finger extension ratios"
)

print(
    "        ↓"
)

print(
    "151 shape features"
)

print(
    "        ↓"
)

print(
    "StandardScaler"
)

print(
    "        ↓"
)

print(
    "RBF SVM"
)

print()
print("=" * 70)