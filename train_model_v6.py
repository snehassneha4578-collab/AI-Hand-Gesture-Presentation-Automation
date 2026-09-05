from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# ============================================================
# V6 ROBUST HAND-GESTURE CLASSIFIER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "gesture_dataset_v2.csv"
MODEL_DIR = BASE_DIR / "trained_model"
REPORT_PATH = BASE_DIR / "dataset" / "v6_training_results.txt"

MODEL_DIR.mkdir(exist_ok=True)


GESTURES = ["fist", "one", "two", "three", "open_palm", "pinch"]

FEATURE_COLUMNS = [
    f"{axis}_{i}"
    for i in range(21)
    for axis in ["x", "y", "z"]
]


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("=" * 75)
print("V6 ROBUST FINGER-STATE CLASSIFIER")
print("=" * 75)

print(f"Dataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")

if "session_id" not in df.columns:
    raise ValueError("session_id column is required for unseen-session evaluation.")

if "label" not in df.columns:
    raise ValueError("label column is missing.")


# ------------------------------------------------------------
# Landmark extraction
# ------------------------------------------------------------

landmark_data = df[FEATURE_COLUMNS].astype(float).values
labels = df["label"].values
sessions = df["session_id"].values


# Shape:
# samples x 21 x 3
landmarks = landmark_data.reshape(-1, 21, 3)


# ------------------------------------------------------------
# V6 FEATURE ENGINEERING
#
# Main idea:
# Do NOT feed 63 raw coordinates directly.
#
# Build compact features describing:
# - palm orientation
# - finger extension
# - finger curl
# - joint angles
# - fingertip distances
# - finger relationships
# - pinch
# - hand openness
# ------------------------------------------------------------

def safe_norm(v):
    return np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8


def normalize(v):
    return v / safe_norm(v)


def angle_between(a, b):
    dot = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)

    cosang = dot / (na * nb + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.arccos(cosang)


def build_features(hand):
    wrist = hand[0]

    # MediaPipe:
    # Thumb: 1,2,3,4
    # Index: 5,6,7,8
    # Middle: 9,10,11,12
    # Ring: 13,14,15,16
    # Pinky: 17,18,19,20

    index_mcp = hand[5]
    middle_mcp = hand[9]
    pinky_mcp = hand[17]

    # --------------------------------------------------------
    # Palm coordinate system
    # --------------------------------------------------------

    x_axis = normalize(index_mcp - pinky_mcp)
    y_temp = middle_mcp - wrist

    # Remove x component from y
    projection = np.sum(y_temp * x_axis, axis=-1, keepdims=True) * x_axis
    y_axis = normalize(y_temp - projection)

    z_axis = normalize(np.cross(x_axis, y_axis))

    # Re-orthogonalize y
    y_axis = normalize(np.cross(z_axis, x_axis))

    # --------------------------------------------------------
    # Wrist-centered + orientation-normalized landmarks
    # --------------------------------------------------------

    centered = hand - wrist

    x_coord = np.sum(centered * x_axis, axis=1)
    y_coord = np.sum(centered * y_axis, axis=1)
    z_coord = np.sum(centered * z_axis, axis=1)

    palm_scale = (
        np.linalg.norm(index_mcp - pinky_mcp, axis=0)
        + np.linalg.norm(middle_mcp - wrist, axis=0)
    ) / 2.0

    palm_scale = float(palm_scale) + 1e-8

    x_coord /= palm_scale
    y_coord /= palm_scale
    z_coord /= palm_scale

    features = []

    # --------------------------------------------------------
    # 1. Compact normalized landmarks
    # --------------------------------------------------------

    for i in range(21):
        features.extend([
            x_coord[i],
            y_coord[i],
            z_coord[i]
        ])

    # --------------------------------------------------------
    # 2. Finger geometric features
    # --------------------------------------------------------

    fingers = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }

    fingertip_indices = [4, 8, 12, 16, 20]

    # --------------------------------------------------------
    # Finger straightness / curl
    # --------------------------------------------------------

    for name, ids in fingers.items():

        mcp, pip, dip, tip = ids

        v1 = hand[pip] - hand[mcp]
        v2 = hand[dip] - hand[pip]
        v3 = hand[tip] - hand[dip]

        a1 = angle_between(v1, v2)
        a2 = angle_between(v2, v3)

        # Straight finger -> angles close to pi
        straightness = (a1 + a2) / (2.0 * np.pi)

        # Tip distance from MCP
        tip_distance = np.linalg.norm(
            hand[tip] - hand[mcp]
        ) / palm_scale

        # Finger path length
        path_length = (
            np.linalg.norm(hand[pip] - hand[mcp]) +
            np.linalg.norm(hand[dip] - hand[pip]) +
            np.linalg.norm(hand[tip] - hand[dip])
        ) / palm_scale

        # Chord / path ratio
        chord = np.linalg.norm(hand[tip] - hand[mcp]) / palm_scale
        curl_ratio = chord / (path_length + 1e-8)

        features.extend([
            straightness,
            tip_distance,
            path_length,
            curl_ratio
        ])

    # --------------------------------------------------------
    # 3. Finger extension state
    # --------------------------------------------------------

    for name, ids in fingers.items():

        mcp, pip, dip, tip = ids

        tip_dist = np.linalg.norm(hand[tip] - wrist)
        pip_dist = np.linalg.norm(hand[pip] - wrist)

        extension_ratio = tip_dist / (pip_dist + 1e-8)

        features.append(extension_ratio)

    # --------------------------------------------------------
    # 4. Fingertip to wrist distances
    # --------------------------------------------------------

    for idx in fingertip_indices:
        features.append(
            np.linalg.norm(hand[idx] - wrist) / palm_scale
        )

    # --------------------------------------------------------
    # 5. Fingertip pairwise distances
    # --------------------------------------------------------

    for i in range(len(fingertip_indices)):
        for j in range(i + 1, len(fingertip_indices)):

            a = fingertip_indices[i]
            b = fingertip_indices[j]

            features.append(
                np.linalg.norm(hand[a] - hand[b]) / palm_scale
            )

    # --------------------------------------------------------
    # 6. Thumb-index pinch
    # --------------------------------------------------------

    pinch_distance = (
        np.linalg.norm(hand[4] - hand[8])
        / palm_scale
    )

    features.append(pinch_distance)

    # --------------------------------------------------------
    # 7. Palm openness
    # --------------------------------------------------------

    palm_width = np.linalg.norm(
        hand[5] - hand[17]
    ) / palm_scale

    palm_height = np.linalg.norm(
        hand[9] - wrist
    ) / palm_scale

    hand_openness = (
        sum(
            np.linalg.norm(hand[idx] - wrist)
            for idx in fingertip_indices
        ) / palm_scale
    )

    features.extend([
        palm_width,
        palm_height,
        hand_openness
    ])

    # --------------------------------------------------------
    # 8. Finger-to-finger relationships
    # --------------------------------------------------------

    for a, b in zip(
        [8, 12, 16, 20],
        [12, 16, 20, 4]
    ):
        features.append(
            np.linalg.norm(hand[a] - hand[b]) / palm_scale
        )

    # --------------------------------------------------------
    # 9. MCP relationships
    # --------------------------------------------------------

    mcp_indices = [5, 9, 13, 17]

    for i in range(len(mcp_indices)):
        for j in range(i + 1, len(mcp_indices)):

            features.append(
                np.linalg.norm(
                    hand[mcp_indices[i]]
                    - hand[mcp_indices[j]]
                ) / palm_scale
            )

    # --------------------------------------------------------
    # 10. Finger joint bending angles
    # --------------------------------------------------------

    for ids in fingers.values():

        mcp, pip, dip, tip = ids

        a = angle_between(
            hand[mcp] - hand[pip],
            hand[dip] - hand[pip]
        )

        b = angle_between(
            hand[pip] - hand[dip],
            hand[tip] - hand[dip]
        )

        features.extend([
            a / np.pi,
            b / np.pi
        ])

    return np.asarray(features, dtype=float)


# ------------------------------------------------------------
# Build feature matrix
# ------------------------------------------------------------

print("\nBuilding V6 features...")

X = np.vstack([
    build_features(hand)
    for hand in landmarks
])

y = labels

print(f"V6 feature count: {X.shape[1]}")
print(f"Feature matrix: {X.shape}")


# ------------------------------------------------------------
# Remove invalid rows
# ------------------------------------------------------------

valid = np.isfinite(X).all(axis=1)

X = X[valid]
y = y[valid]
sessions = sessions[valid]

print(f"Valid samples: {len(X)}")


# ------------------------------------------------------------
# Unseen-session split
#
# Hold out the newest session for every gesture.
# ------------------------------------------------------------

session_info = (
    df.loc[valid, ["label", "session_id"]]
    .drop_duplicates()
)

test_sessions = {}

for gesture in GESTURES:

    gesture_sessions = (
        session_info[
            session_info["label"] == gesture
        ]["session_id"]
        .tolist()
    )

    if not gesture_sessions:
        raise ValueError(
            f"No session found for gesture: {gesture}"
        )

    # Last session = newest session
    test_sessions[gesture] = gesture_sessions[-1]


test_session_ids = set(test_sessions.values())

test_mask = np.array([
    s in test_session_ids
    for s in sessions
])

train_mask = ~test_mask


X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print("\nTEST SESSIONS")

for gesture, session_id in test_sessions.items():
    print(f"{gesture}: {session_id}")

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Training sessions: {len(set(sessions[train_mask]))}")
print(f"Testing sessions: {len(set(sessions[test_mask]))}")


# ------------------------------------------------------------
# Class balancing
# ------------------------------------------------------------

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# ------------------------------------------------------------
# Model 1: SVM
# ------------------------------------------------------------

print("\nTraining V6 SVM...")

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
])

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print(
    f"V6 SVM accuracy: "
    f"{svm_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# Model 2: Random Forest
# ------------------------------------------------------------

print("\nTraining V6 Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    f"V6 Random Forest accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# Model 3: Extra Trees
# ------------------------------------------------------------

print("\nTraining V6 Extra Trees...")

et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

et_model.fit(
    X_train,
    y_train
)

et_predictions = et_model.predict(X_test)

et_accuracy = accuracy_score(
    y_test,
    et_predictions
)

print(
    f"V6 Extra Trees accuracy: "
    f"{et_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# Select best model
# ------------------------------------------------------------

models = {
    "SVM": (svm_accuracy, svm_model, svm_predictions),
    "Random Forest": (rf_accuracy, rf_model, rf_predictions),
    "Extra Trees": (et_accuracy, et_model, et_predictions)
}

best_name = max(
    models,
    key=lambda name: models[name][0]
)

best_accuracy, best_model, best_predictions = models[best_name]


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report = classification_report(
    y_test,
    best_predictions,
    labels=GESTURES,
    target_names=GESTURES,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=GESTURES
)


print("\n" + "=" * 75)
print("V6 RESULTS")
print("=" * 75)

print(f"SVM:          {svm_accuracy * 100:.2f}%")
print(f"Random Forest:{rf_accuracy * 100:.2f}%")
print(f"Extra Trees:  {et_accuracy * 100:.2f}%")
print(f"BEST MODEL:   {best_name}")
print(f"BEST ACCURACY:{best_accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(report)

print("CONFUSION MATRIX")
print(cm)


# ------------------------------------------------------------
# Per gesture accuracy
# ------------------------------------------------------------

print("\nPER-GESTURE ACCURACY")

per_gesture = {}

for i, gesture in enumerate(GESTURES):

    total = cm[i].sum()

    if total > 0:
        acc = cm[i, i] / total
    else:
        acc = 0

    per_gesture[gesture] = acc

    print(
        f"{gesture}: "
        f"{acc * 100:.2f}%"
    )


# ------------------------------------------------------------
# Prediction distribution
# ------------------------------------------------------------

print("\nPREDICTION DISTRIBUTION")

prediction_counts = pd.Series(
    best_predictions
).value_counts()

for gesture in GESTURES:

    count = int(
        prediction_counts.get(gesture, 0)
    )

    percentage = (
        count / len(best_predictions) * 100
    )

    print(
        f"{gesture}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ------------------------------------------------------------
# Save best model
# ------------------------------------------------------------

if best_name == "SVM":

    joblib.dump(
        svm_model,
        MODEL_DIR / "gesture_svm_v6.pkl"
    )

elif best_name == "Random Forest":

    joblib.dump(
        rf_model,
        MODEL_DIR / "gesture_rf_v6.pkl"
    )

else:

    joblib.dump(
        et_model,
        MODEL_DIR / "gesture_et_v6.pkl"
    )


# Save feature information
feature_info = {
    "feature_count": int(X.shape[1]),
    "feature_type": "V6 normalized hand geometry and finger-state features",
    "gestures": GESTURES
}

joblib.dump(
    feature_info,
    MODEL_DIR / "v6_feature_info.pkl"
)


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("V6 ROBUST FINGER-STATE CLASSIFIER\n")
    f.write("=" * 75 + "\n")

    f.write(f"Dataset rows: {len(df)}\n")
    f.write(f"Dataset columns: {len(df.columns)}\n")
    f.write(f"Sessions: {len(set(sessions))}\n")

    f.write(f"V6 features: {X.shape[1]}\n")

    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")

    f.write(
        f"SVM accuracy: "
        f"{svm_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Random Forest accuracy: "
        f"{rf_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Extra Trees accuracy: "
        f"{et_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Best model: {best_name}\n"
    )

    f.write(
        f"Best accuracy: "
        f"{best_accuracy * 100:.2f}%\n"
    )

    f.write("\nTEST SESSIONS\n")

    for gesture, session_id in test_sessions.items():
        f.write(
            f"{gesture}: {session_id}\n"
        )

    f.write("\nCLASSIFICATION REPORT\n")
    f.write(report)

    f.write("\nCONFUSION MATRIX\n")

    f.write(
        pd.DataFrame(
            cm,
            index=[
                f"Actual_{g}"
                for g in GESTURES
            ],
            columns=[
                f"Pred_{g}"
                for g in GESTURES
            ]
        ).to_string()
    )

    f.write("\n\nPER-GESTURE ACCURACY\n")

    for gesture, acc in per_gesture.items():

        f.write(
            f"{gesture}: "
            f"{acc * 100:.2f}%\n"
        )

    f.write("\n\nINTERPRETATION\n")

    f.write(
        "V6 evaluates compact finger-state and geometric "
        "features using an unseen-session split. "
        "The newest session for every gesture is held out "
        "from training. This is intended to measure "
        "generalization to a new collection session.\n"
    )

    f.write(
        "\nDo not compare this directly with the 92.36% "
        "random row-split result.\n"
    )

    f.write(
        "\nMODEL FILES\n"
    )

    if best_name == "SVM":
        f.write(
            str(
                MODEL_DIR /
                "gesture_svm_v6.pkl"
            )
        )

    elif best_name == "Random Forest":
        f.write(
            str(
                MODEL_DIR /
                "gesture_rf_v6.pkl"
            )
        )

    else:
        f.write(
            str(
                MODEL_DIR /
                "gesture_et_v6.pkl"
            )
        )


print("\n" + "=" * 75)
print("V6 TRAINING COMPLETE")
print("=" * 75)

print(
    f"Best model: {best_name}"
)

print(
    f"Best unseen-session accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"\nReport saved:\n{REPORT_PATH}"
)

print("\nNext step:")
print(
    "Evaluate the best V6 model with a completely "
    "new webcam session before using it for the "
    "real-time presentation controller."
)
