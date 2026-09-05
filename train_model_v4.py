from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# V4 GEOMETRY-BASED GESTURE CLASSIFIER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "gesture_dataset_v2.csv"

MODEL_DIR = BASE_DIR / "trained_model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "gesture_svm_v4.pkl"
SCALER_PATH = MODEL_DIR / "scaler_v4.pkl"

FEATURE_OUTPUT = (
    BASE_DIR / "dataset" / "gesture_dataset_v4_features.csv"
)

REPORT_PATH = (
    BASE_DIR / "dataset" / "v4_training_results.txt"
)

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]

# ============================================================
# LANDMARK DEFINITIONS
# ============================================================

WRIST = 0

THUMB_CMC = 1
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

FINGER_TIPS = [
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP
]

FINGER_MCPS = [
    THUMB_MCP,
    INDEX_MCP,
    MIDDLE_MCP,
    RING_MCP,
    PINKY_MCP
]

FINGER_PIPS = [
    THUMB_IP,
    INDEX_PIP,
    MIDDLE_PIP,
    RING_PIP,
    PINKY_PIP
]

FINGER_DIPS = [
    None,
    INDEX_DIP,
    MIDDLE_DIP,
    RING_DIP,
    PINKY_DIP
]


# ============================================================
# RAW FEATURE COLUMNS
# ============================================================

RAW_COLUMNS = []

for i in range(21):
    RAW_COLUMNS.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def angle(a, b, c):
    """
    Angle ABC.
    """
    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.arccos(cosine))


def safe_ratio(a, b):
    if abs(b) < 1e-8:
        return 0.0

    return float(a / b)


# ============================================================
# LANDMARK NORMALIZATION
# ============================================================

def normalize_landmarks(row):
    landmarks = np.array([
        [
            row[f"x_{i}"],
            row[f"y_{i}"],
            row[f"z_{i}"]
        ]
        for i in range(21)
    ], dtype=float)

    # Wrist becomes origin
    landmarks = landmarks - landmarks[WRIST]

    # Hand scale
    scale = np.linalg.norm(
        landmarks[MIDDLE_MCP]
    )

    if scale < 1e-8:

        scale = np.mean([
            np.linalg.norm(
                landmarks[mcp]
            )
            for mcp in FINGER_MCPS
            if mcp != THUMB_MCP
        ])

    if scale < 1e-8:
        scale = 1.0

    landmarks = landmarks / scale

    return landmarks


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(row):

    lm = normalize_landmarks(row)

    features = []

    # --------------------------------------------------------
    # 1. Normalized landmark coordinates
    # --------------------------------------------------------

    features.extend(
        lm.flatten().tolist()
    )

    # --------------------------------------------------------
    # 2. Wrist-to-landmark distances
    # --------------------------------------------------------

    for i in range(1, 21):

        features.append(
            distance(
                lm[WRIST],
                lm[i]
            )
        )

    # --------------------------------------------------------
    # 3. Fingertip-to-wrist distances
    # --------------------------------------------------------

    for tip in FINGER_TIPS:

        features.append(
            distance(
                lm[WRIST],
                lm[tip]
            )
        )

    # --------------------------------------------------------
    # 4. Fingertip-to-middle-finger-tip distances
    # --------------------------------------------------------

    reference_tip = MIDDLE_TIP

    for tip in FINGER_TIPS:

        features.append(
            distance(
                lm[tip],
                lm[reference_tip]
            )
        )

    # --------------------------------------------------------
    # 5. Pairwise fingertip distances
    # --------------------------------------------------------

    for i in range(len(FINGER_TIPS)):

        for j in range(
            i + 1,
            len(FINGER_TIPS)
        ):

            features.append(
                distance(
                    lm[FINGER_TIPS[i]],
                    lm[FINGER_TIPS[j]]
                )
            )

    # --------------------------------------------------------
    # 6. Finger bone lengths
    # --------------------------------------------------------

    bones = [
        (THUMB_CMC, THUMB_MCP),
        (THUMB_MCP, THUMB_IP),
        (THUMB_IP, THUMB_TIP),

        (INDEX_MCP, INDEX_PIP),
        (INDEX_PIP, INDEX_DIP),
        (INDEX_DIP, INDEX_TIP),

        (MIDDLE_MCP, MIDDLE_PIP),
        (MIDDLE_PIP, MIDDLE_DIP),
        (MIDDLE_DIP, MIDDLE_TIP),

        (RING_MCP, RING_PIP),
        (RING_PIP, RING_DIP),
        (RING_DIP, RING_TIP),

        (PINKY_MCP, PINKY_PIP),
        (PINKY_PIP, PINKY_DIP),
        (PINKY_DIP, PINKY_TIP)
    ]

    for a, b in bones:

        features.append(
            distance(
                lm[a],
                lm[b]
            )
        )

    # --------------------------------------------------------
    # 7. Finger joint angles
    # --------------------------------------------------------

    angle_triplets = [
        (THUMB_CMC, THUMB_MCP, THUMB_IP),
        (THUMB_MCP, THUMB_IP, THUMB_TIP),

        (INDEX_MCP, INDEX_PIP, INDEX_DIP),
        (INDEX_PIP, INDEX_DIP, INDEX_TIP),

        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
        (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),

        (RING_MCP, RING_PIP, RING_DIP),
        (RING_PIP, RING_DIP, RING_TIP),

        (PINKY_MCP, PINKY_PIP, PINKY_DIP),
        (PINKY_PIP, PINKY_DIP, PINKY_TIP)
    ]

    for a, b, c in angle_triplets:

        features.append(
            angle(
                lm[a],
                lm[b],
                lm[c]
            )
        )

    # --------------------------------------------------------
    # 8. Finger extension ratios
    # --------------------------------------------------------

    for idx in range(5):

        mcp = FINGER_MCPS[idx]
        pip = FINGER_PIPS[idx]
        tip = FINGER_TIPS[idx]

        wrist_to_tip = distance(
            lm[WRIST],
            lm[tip]
        )

        wrist_to_mcp = distance(
            lm[WRIST],
            lm[mcp]
        )

        features.append(
            safe_ratio(
                wrist_to_tip,
                wrist_to_mcp
            )
        )

    # --------------------------------------------------------
    # 9. Fingertip height relative to MCP
    # --------------------------------------------------------

    for idx in range(5):

        mcp = FINGER_MCPS[idx]
        tip = FINGER_TIPS[idx]

        features.append(
            float(
                lm[tip][1] -
                lm[mcp][1]
            )
        )

    # --------------------------------------------------------
    # 10. Fingertip x-position relative to MCP
    # --------------------------------------------------------

    for idx in range(5):

        mcp = FINGER_MCPS[idx]
        tip = FINGER_TIPS[idx]

        features.append(
            float(
                lm[tip][0] -
                lm[mcp][0]
            )
        )

    # --------------------------------------------------------
    # 11. Palm geometry
    # --------------------------------------------------------

    palm_pairs = [
        (WRIST, INDEX_MCP),
        (WRIST, MIDDLE_MCP),
        (WRIST, RING_MCP),
        (WRIST, PINKY_MCP),

        (INDEX_MCP, MIDDLE_MCP),
        (MIDDLE_MCP, RING_MCP),
        (RING_MCP, PINKY_MCP),

        (INDEX_MCP, RING_MCP),
        (INDEX_MCP, PINKY_MCP),
        (MIDDLE_MCP, PINKY_MCP)
    ]

    for a, b in palm_pairs:

        features.append(
            distance(
                lm[a],
                lm[b]
            )
        )

    return features


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("V4 GEOMETRY-BASED GESTURE CLASSIFIER")
print("=" * 70)

print(f"\nDataset:")
print(DATASET_PATH)

if not DATASET_PATH.exists():

    print("\nERROR: Dataset not found.")
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Sessions: {df['session_id'].nunique()}")

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

missing = df[RAW_COLUMNS].isnull().sum().sum()

values = df[RAW_COLUMNS].to_numpy(
    dtype=float
)

non_finite = np.sum(
    ~np.isfinite(values)
)

print(f"Missing values: {missing}")
print(f"Non-finite values: {non_finite}")

if missing > 0 or non_finite > 0:

    print("\nERROR: Invalid feature values.")
    raise SystemExit(1)

print("\nGesture distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# GENERATE V4 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING V4 GEOMETRY FEATURES")
print("=" * 70)

feature_rows = []

for index, row in df.iterrows():

    try:

        feature_rows.append(
            extract_features(row)
        )

    except Exception as error:

        print(
            f"Feature extraction error at row "
            f"{index}: {error}"
        )

        raise

X = np.asarray(
    feature_rows,
    dtype=float
)

y = df["label"].to_numpy()

print(
    f"\nV4 feature count: {X.shape[1]}"
)

print(
    f"Feature matrix shape: {X.shape}"
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

if not np.all(np.isfinite(X)):

    print(
        "\nERROR: V4 features contain "
        "non-finite values."
    )

    raise SystemExit(1)

print(
    "All V4 features are finite."
)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

feature_names = [
    f"feature_{i}"
    for i in range(X.shape[1])
]

feature_df = pd.DataFrame(
    X,
    columns=feature_names
)

feature_df["label"] = y
feature_df["session_id"] = df["session_id"].values

feature_df.to_csv(
    FEATURE_OUTPUT,
    index=False
)

print(
    f"\nSaved V4 features:"
)
print(FEATURE_OUTPUT)


# ============================================================
# SESSION-AWARE SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SESSION-AWARE TRAIN / TEST SPLIT")
print("=" * 70)

session_table = (
    df.groupby(
        ["session_id", "label"]
    )
    .size()
    .reset_index(
        name="samples"
    )
)

print("\nAvailable sessions:")

print(
    session_table.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Select one test session for every gesture.
#
# We deliberately use one complete session per class.
# This prevents rows from the same recording session from
# appearing in both train and test.
# ------------------------------------------------------------

test_sessions = {}

for gesture in GESTURES:

    gesture_sessions = (
        session_table[
            session_table["label"] == gesture
        ]
        .sort_values(
            "session_id"
        )
        ["session_id"]
        .tolist()
    )

    if len(gesture_sessions) < 2:

        print(
            f"\nERROR: Gesture '{gesture}' "
            f"does not have at least two sessions."
        )

        raise SystemExit(1)

    # Use the newest available session
    # for testing.
    test_sessions[gesture] = (
        gesture_sessions[-1]
    )


print("\nSelected test sessions:")

for gesture, session_id in test_sessions.items():

    print(
        f"{gesture:12s} -> {session_id}"
    )

test_mask = np.array([
    test_sessions[label] == session_id
    for label, session_id
    in zip(
        df["label"],
        df["session_id"]
    )
])

train_mask = ~test_mask

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)

print(
    f"Training sessions: "
    f"{df.loc[train_mask, 'session_id'].nunique()}"
)

print(
    f"Testing sessions: "
    f"{df.loc[test_mask, 'session_id'].nunique()}"
)

print("\nTraining distribution:")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTesting distribution:")

print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "StandardScaler fitted using TRAINING "
    "sessions only."
)


# ============================================================
# TRAIN SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING V4 SVM")
print("=" * 70)

print(
    "\nModel:"
)
print(
    "RBF SVM"
)
print(
    "C = 10"
)
print(
    "gamma = scale"
)
print(
    "probability = True"
)

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "\nTraining complete."
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("V4 UNSEEN-SESSION EVALUATION")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nV4 Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=GESTURES,
    target_names=GESTURES,
    zero_division=0
)

print(report)

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=GESTURES
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
    cm_df.to_string()
)


# ============================================================
# PER-GESTURE ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-GESTURE ACCURACY")
print("=" * 70)

per_gesture = {}

for gesture in GESTURES:

    mask = y_test == gesture

    total = np.sum(mask)

    correct = np.sum(
        y_pred[mask] == gesture
    )

    if total > 0:

        acc = (
            correct /
            total *
            100
        )

    else:

        acc = 0.0

    per_gesture[gesture] = acc

    print(
        f"{gesture:12s}: "
        f"{acc:.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_counts = (
    pd.Series(y_pred)
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
        count /
        len(y_pred) *
        100
    )

    print(
        f"{gesture:12s}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# MODEL SAVING
# ============================================================

print("\n" + "=" * 70)
print("SAVING V4 MODEL")
print("=" * 70)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    f"\nModel saved:"
)
print(MODEL_PATH)

print(
    f"\nScaler saved:"
)
print(SCALER_PATH)


# ============================================================
# REPORT
# ============================================================

report_lines = []

report_lines.append(
    "V4 GEOMETRY-BASED GESTURE CLASSIFIER"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    f"Dataset rows: {len(df)}"
)

report_lines.append(
    f"Dataset columns: {len(df.columns)}"
)

report_lines.append(
    f"Sessions: {df['session_id'].nunique()}"
)

report_lines.append(
    f"V4 features: {X.shape[1]}"
)

report_lines.append(
    f"Training samples: {len(X_train)}"
)

report_lines.append(
    f"Testing samples: {len(X_test)}"
)

report_lines.append(
    f"V4 Accuracy: {accuracy * 100:.2f}%"
)

report_lines.append(
    "\nTEST SESSIONS"
)

for gesture, session_id in test_sessions.items():

    report_lines.append(
        f"{gesture}: {session_id}"
    )

report_lines.append(
    "\nCLASSIFICATION REPORT"
)

report_lines.append(
    report
)

report_lines.append(
    "\nCONFUSION MATRIX"
)

report_lines.append(
    cm_df.to_string()
)

report_lines.append(
    "\nPER-GESTURE ACCURACY"
)

for gesture, acc in per_gesture.items():

    report_lines.append(
        f"{gesture}: {acc:.2f}%"
    )

report_lines.append(
    "\nPREDICTION DISTRIBUTION"
)

for gesture in GESTURES:

    count = int(
        prediction_counts.get(
            gesture,
            0
        )
    )

    percentage = (
        count /
        len(y_pred) *
        100
    )

    report_lines.append(
        f"{gesture}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

report_lines.append(
    "\nMODEL FILE"
)

report_lines.append(
    str(MODEL_PATH)
)

report_lines.append(
    "\nSCALER FILE"
)

report_lines.append(
    str(SCALER_PATH)
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print(
    f"\nTraining report saved:"
)

print(REPORT_PATH)

print("\n" + "=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "This V4 result is an UNSEEN-SESSION "
    "evaluation."
)

print(
    "Do not compare it directly with the "
    "92.36% random-split result."
)

print(
    "\nNext step after reviewing V4:"
)

print(
    "Run the V4 model on a completely new "
    "live webcam session."
)
