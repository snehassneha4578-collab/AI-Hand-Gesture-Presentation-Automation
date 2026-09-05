from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# V5 CONTROLLED GEOMETRY CLASSIFIER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "gesture_dataset_v2.csv"

MODEL_DIR = BASE_DIR / "trained_model"
MODEL_DIR.mkdir(exist_ok=True)

SVM_MODEL_PATH = MODEL_DIR / "gesture_svm_v5.pkl"
SVM_SCALER_PATH = MODEL_DIR / "scaler_v5.pkl"

RF_MODEL_PATH = MODEL_DIR / "gesture_rf_v5.pkl"

FEATURE_OUTPUT = BASE_DIR / "dataset" / "gesture_dataset_v5_features.csv"
REPORT_PATH = BASE_DIR / "dataset" / "v5_training_results.txt"

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]


# ============================================================
# LANDMARK INDICES
# ============================================================

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


FINGER_DATA = [
    ("thumb", THUMB_MCP, THUMB_IP, None, THUMB_TIP),
    ("index", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    ("ring", RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    ("pinky", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
]


RAW_COLUMNS = []

for i in range(21):
    RAW_COLUMNS.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


# ============================================================
# BASIC GEOMETRY
# ============================================================

def dist(a, b):
    return float(np.linalg.norm(a - b))


def safe_angle(a, b, c):
    ba = a - b
    bc = c - b

    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)

    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    value = np.dot(ba, bc) / (n1 * n2)
    value = np.clip(value, -1.0, 1.0)

    return float(np.arccos(value))


def safe_ratio(a, b):
    if abs(b) < 1e-9:
        return 0.0

    return float(a / b)


# ============================================================
# LOAD LANDMARKS
# ============================================================

def get_landmarks(row):

    return np.array([
        [
            float(row[f"x_{i}"]),
            float(row[f"y_{i}"]),
            float(row[f"z_{i}"])
        ]
        for i in range(21)
    ], dtype=float)


# ============================================================
# V5 NORMALIZATION
# ============================================================

def normalize_landmarks(row):

    lm = get_landmarks(row)

    # --------------------------------------------------------
    # Step 1: wrist becomes origin
    # --------------------------------------------------------

    lm = lm - lm[WRIST]

    # --------------------------------------------------------
    # Step 2: hand scale
    # --------------------------------------------------------

    scale = np.linalg.norm(lm[MIDDLE_MCP])

    if scale < 1e-8:

        scale = np.mean([
            np.linalg.norm(lm[i])
            for i in [
                INDEX_MCP,
                MIDDLE_MCP,
                RING_MCP,
                PINKY_MCP
            ]
        ])

    if scale < 1e-8:
        scale = 1.0

    lm = lm / scale

    # --------------------------------------------------------
    # Step 3: rotation normalization
    #
    # Align wrist -> middle MCP with the Y axis.
    # This reduces variation caused by hand rotation.
    # --------------------------------------------------------

    reference = lm[MIDDLE_MCP].copy()

    angle_xy = np.arctan2(
        reference[0],
        reference[1]
    )

    cos_a = np.cos(-angle_xy)
    sin_a = np.sin(-angle_xy)

    rotation_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    lm = lm @ rotation_z.T

    # --------------------------------------------------------
    # Additional 3D orientation normalization.
    #
    # Use wrist -> middle MCP vector to approximately align
    # the hand depth direction.
    # --------------------------------------------------------

    direction = lm[MIDDLE_MCP]

    xy_norm = np.sqrt(
        direction[0] ** 2 +
        direction[1] ** 2
    )

    if xy_norm > 1e-8:

        pitch = np.arctan2(
            direction[2],
            xy_norm
        )

        cos_p = np.cos(-pitch)
        sin_p = np.sin(-pitch)

        rotation_x = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])

        lm = lm @ rotation_x.T

    return lm


# ============================================================
# V5 FEATURE EXTRACTION
# ============================================================

def extract_v5_features(row):

    lm = normalize_landmarks(row)

    features = []

    # ========================================================
    # 1. Compact normalized landmark representation
    #
    # Keep coordinates, but use normalized coordinates.
    # ========================================================

    for i in range(21):

        features.extend([
            lm[i][0],
            lm[i][1],
            lm[i][2]
        ])

    # ========================================================
    # 2. Finger joint angles
    # ========================================================

    angle_sets = [
        # thumb
        (THUMB_MCP, THUMB_IP, THUMB_TIP),

        # index
        (INDEX_MCP, INDEX_PIP, INDEX_DIP),
        (INDEX_PIP, INDEX_DIP, INDEX_TIP),

        # middle
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
        (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),

        # ring
        (RING_MCP, RING_PIP, RING_DIP),
        (RING_PIP, RING_DIP, RING_TIP),

        # pinky
        (PINKY_MCP, PINKY_PIP, PINKY_DIP),
        (PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]

    for a, b, c in angle_sets:

        features.append(
            safe_angle(
                lm[a],
                lm[b],
                lm[c]
            )
        )

    # ========================================================
    # 3. Finger curl / extension measurements
    # ========================================================

    for name, mcp, pip, dip, tip in FINGER_DATA:

        wrist_to_tip = dist(
            lm[WRIST],
            lm[tip]
        )

        wrist_to_mcp = dist(
            lm[WRIST],
            lm[mcp]
        )

        features.append(
            safe_ratio(
                wrist_to_tip,
                wrist_to_mcp
            )
        )

        # fingertip to MCP distance
        features.append(
            dist(
                lm[mcp],
                lm[tip]
            )
        )

        # fingertip height relative to MCP
        features.append(
            lm[tip][1] - lm[mcp][1]
        )

        # fingertip horizontal displacement
        features.append(
            lm[tip][0] - lm[mcp][0]
        )

    # ========================================================
    # 4. Fingertip distances
    # ========================================================

    tips = [
        THUMB_TIP,
        INDEX_TIP,
        MIDDLE_TIP,
        RING_TIP,
        PINKY_TIP
    ]

    for i in range(len(tips)):

        for j in range(i + 1, len(tips)):

            features.append(
                dist(
                    lm[tips[i]],
                    lm[tips[j]]
                )
            )

    # ========================================================
    # 5. Fingertip-to-wrist distances
    # ========================================================

    for tip in tips:

        features.append(
            dist(
                lm[WRIST],
                lm[tip]
            )
        )

    # ========================================================
    # 6. Palm geometry
    # ========================================================

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
            dist(
                lm[a],
                lm[b]
            )
        )

    # ========================================================
    # 7. Relative finger-to-finger relationships
    # ========================================================

    for tip in [
        INDEX_TIP,
        MIDDLE_TIP,
        RING_TIP,
        PINKY_TIP
    ]:

        features.append(
            safe_ratio(
                dist(
                    lm[WRIST],
                    lm[tip]
                ),
                dist(
                    lm[WRIST],
                    lm[MIDDLE_TIP]
                )
            )
        )

    # ========================================================
    # 8. Finger bending angles at MCP
    # ========================================================

    mcp_angle_sets = [
        (WRIST, INDEX_MCP, INDEX_PIP),
        (WRIST, MIDDLE_MCP, MIDDLE_PIP),
        (WRIST, RING_MCP, RING_PIP),
        (WRIST, PINKY_MCP, PINKY_PIP),
    ]

    for a, b, c in mcp_angle_sets:

        features.append(
            safe_angle(
                lm[a],
                lm[b],
                lm[c]
            )
        )

    return features


# ============================================================
# START
# ============================================================

print("=" * 75)
print("V5 CONTROLLED GEOMETRY CLASSIFIER")
print("=" * 75)

print("\nDataset:")
print(DATASET_PATH)

if not DATASET_PATH.exists():

    print("\nERROR: Dataset not found.")
    raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET_PATH)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Sessions: {df['session_id'].nunique()}")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("DATASET VALIDATION")
print("=" * 75)

missing = df[RAW_COLUMNS].isnull().sum().sum()

raw_values = df[RAW_COLUMNS].to_numpy(
    dtype=float
)

nonfinite = np.sum(
    ~np.isfinite(raw_values)
)

print(f"Missing values: {missing}")
print(f"Non-finite values: {nonfinite}")

if missing > 0 or nonfinite > 0:

    print("\nERROR: Invalid dataset.")
    raise SystemExit(1)

print("\nGesture distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 75)
print("GENERATING V5 FEATURES")
print("=" * 75)

feature_rows = []

for index, row in df.iterrows():

    feature_rows.append(
        extract_v5_features(row)
    )

X = np.asarray(
    feature_rows,
    dtype=float
)

y = df["label"].to_numpy()

print(
    f"\nV5 feature count: {X.shape[1]}"
)

print(
    f"Feature matrix: {X.shape}"
)

if not np.all(np.isfinite(X)):

    print(
        "\nERROR: V5 features contain "
        "non-finite values."
    )

    raise SystemExit(1)

print("All V5 features are finite.")


# ============================================================
# SAVE FEATURES
# ============================================================

feature_names = [
    f"v5_feature_{i}"
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
    f"\nFeature dataset saved:"
)

print(FEATURE_OUTPUT)


# ============================================================
# SESSION-AWARE SPLIT
# ============================================================

print("\n" + "=" * 75)
print("UNSEEN-SESSION SPLIT")
print("=" * 75)

session_table = (
    df.groupby(
        ["session_id", "label"]
    )
    .size()
    .reset_index(
        name="samples"
    )
)

test_sessions = {}

for gesture in GESTURES:

    sessions = (
        session_table[
            session_table["label"] == gesture
        ]
        .sort_values("session_id")
        ["session_id"]
        .tolist()
    )

    if len(sessions) < 2:

        print(
            f"\nERROR: '{gesture}' needs "
            f"at least 2 sessions."
        )

        raise SystemExit(1)

    # Newest session becomes test session.
    test_sessions[gesture] = sessions[-1]


print("\nTEST SESSIONS:")

for gesture in GESTURES:

    print(
        f"{gesture:12s} -> "
        f"{test_sessions[gesture]}"
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


# ============================================================
# SCALE FOR SVM
# ============================================================

print("\n" + "=" * 75)
print("SVM TRAINING")
print("=" * 75)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# SVM
# ============================================================

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

svm.fit(
    X_train_scaled,
    y_train
)

svm_pred = svm.predict(
    X_test_scaled
)

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST TRAINING")
print("=" * 75)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("V5 MODEL COMPARISON")
print("=" * 75)

print(
    f"\nSVM accuracy:          "
    f"{svm_accuracy * 100:.2f}%"
)

print(
    f"Random Forest accuracy:"
    f" {rf_accuracy * 100:.2f}%"
)


if rf_accuracy > svm_accuracy:

    best_name = "Random Forest"
    best_model = rf
    best_pred = rf_pred

else:

    best_name = "SVM"
    best_model = svm
    best_pred = svm_pred


best_accuracy = max(
    svm_accuracy,
    rf_accuracy
)

print(
    f"\nBEST MODEL: {best_name}"
)

print(
    f"BEST ACCURACY: "
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# BEST MODEL REPORT
# ============================================================

print("\n" + "=" * 75)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 75)

best_report = classification_report(
    y_test,
    best_pred,
    labels=GESTURES,
    target_names=GESTURES,
    zero_division=0
)

print(best_report)


print("CONFUSION MATRIX:")

best_cm = confusion_matrix(
    y_test,
    best_pred,
    labels=GESTURES
)

best_cm_df = pd.DataFrame(
    best_cm,
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
    best_cm_df.to_string()
)


# ============================================================
# PER-GESTURE ACCURACY
# ============================================================

print("\n" + "=" * 75)
print("PER-GESTURE ACCURACY")
print("=" * 75)

per_gesture = {}

for gesture in GESTURES:

    mask = y_test == gesture

    total = np.sum(mask)

    correct = np.sum(
        best_pred[mask] == gesture
    )

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    per_gesture[gesture] = accuracy

    print(
        f"{gesture:12s}: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION DISTRIBUTION")
print("=" * 75)

pred_counts = pd.Series(
    best_pred
).value_counts()

for gesture in GESTURES:

    count = int(
        pred_counts.get(
            gesture,
            0
        )
    )

    percentage = (
        count /
        len(best_pred) *
        100
    )

    print(
        f"{gesture:12s}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE MODELS
# ============================================================

print("\n" + "=" * 75)
print("SAVING V5 MODELS")
print("=" * 75)

joblib.dump(
    svm,
    SVM_MODEL_PATH
)

joblib.dump(
    scaler,
    SVM_SCALER_PATH
)

joblib.dump(
    rf,
    RF_MODEL_PATH
)

print(
    f"\nSVM model:"
)

print(SVM_MODEL_PATH)

print(
    f"\nSVM scaler:"
)

print(SVM_SCALER_PATH)

print(
    f"\nRandom Forest:"
)

print(RF_MODEL_PATH)


# ============================================================
# SAVE REPORT
# ============================================================

lines = []

lines.append(
    "V5 CONTROLLED GEOMETRY CLASSIFIER"
)

lines.append(
    "=" * 75
)

lines.append(
    f"Dataset rows: {len(df)}"
)

lines.append(
    f"Dataset columns: {len(df.columns)}"
)

lines.append(
    f"Sessions: {df['session_id'].nunique()}"
)

lines.append(
    f"V5 features: {X.shape[1]}"
)

lines.append(
    f"Training samples: {len(X_train)}"
)

lines.append(
    f"Testing samples: {len(X_test)}"
)

lines.append(
    f"SVM accuracy: {svm_accuracy * 100:.2f}%"
)

lines.append(
    f"Random Forest accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)

lines.append(
    f"Best model: {best_name}"
)

lines.append(
    f"Best accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

lines.append(
    "\nTEST SESSIONS"
)

for gesture in GESTURES:

    lines.append(
        f"{gesture}: "
        f"{test_sessions[gesture]}"
    )

lines.append(
    "\nCLASSIFICATION REPORT"
)

lines.append(
    best_report
)

lines.append(
    "\nCONFUSION MATRIX"
)

lines.append(
    best_cm_df.to_string()
)

lines.append(
    "\nPER-GESTURE ACCURACY"
)

for gesture in GESTURES:

    lines.append(
        f"{gesture}: "
        f"{per_gesture[gesture]:.2f}%"
    )

lines.append(
    "\nINTERPRETATION"
)

if best_accuracy >= 80:

    lines.append(
        "V5 shows strong unseen-session "
        "generalization."
    )

elif best_accuracy >= 60:

    lines.append(
        "V5 shows meaningful improvement, "
        "but further robustness work is "
        "recommended."
    )

elif best_accuracy >= 40:

    lines.append(
        "V5 is better than the previous "
        "geometry models but generalization "
        "remains limited."
    )

else:

    lines.append(
        "V5 still has poor unseen-session "
        "generalization. Do not use this "
        "accuracy as the final project claim."
    )

lines.append(
    "\nMODEL FILES"
)

lines.append(
    str(SVM_MODEL_PATH)
)

lines.append(
    str(SVM_SCALER_PATH)
)

lines.append(
    str(RF_MODEL_PATH)
)

REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print(
    f"\nReport saved:"
)

print(REPORT_PATH)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("V5 TRAINING COMPLETE")
print("=" * 75)

print(
    f"\nSVM: "
    f"{svm_accuracy * 100:.2f}%"
)

print(
    f"Random Forest: "
    f"{rf_accuracy * 100:.2f}%"
)

print(
    f"Best: {best_name} "
    f"({best_accuracy * 100:.2f}%)"
)

print(
    "\nIMPORTANT:"
)

print(
    "This is an UNSEEN-SESSION evaluation."
)

print(
    "Do not compare it directly with the "
    "92.36% random row-split result."
)

print(
    "\nNext step: evaluate the best V5 model "
    "with a completely new webcam session."
)
