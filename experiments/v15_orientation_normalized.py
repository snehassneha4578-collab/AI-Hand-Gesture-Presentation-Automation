import pandas as pd
import numpy as np
import os

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")
REPORT = os.path.join(BASE_DIR, "dataset", "v15_orientation_normalized_results.txt")

GESTURES = ["fist", "one", "open_palm", "pinch", "three", "two"]


def landmark(row, i):
    return np.array([
        float(row[f"x_{i}"]),
        float(row[f"y_{i}"]),
        float(row[f"z_{i}"])
    ], dtype=float)


def distance(a, b):
    return np.linalg.norm(a - b)


def normalize_hand(row):

    pts = np.array([
        landmark(row, i)
        for i in range(21)
    ])

    # Translation invariance
    pts = pts - pts[0]

    # Palm reference axes
    middle_mcp = pts[9]
    index_mcp = pts[5]
    pinky_mcp = pts[17]

    y_axis = middle_mcp
    y_norm = np.linalg.norm(y_axis)

    if y_norm < 1e-9:
        return None

    y_axis = y_axis / y_norm

    x_axis = pinky_mcp - index_mcp
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis

    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-9:
        return None

    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)

    if z_norm < 1e-9:
        return None

    z_axis = z_axis / z_norm

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    transformed = []

    for point in pts:
        transformed.append([
            np.dot(point, x_axis),
            np.dot(point, y_axis),
            np.dot(point, z_axis)
        ])

    transformed = np.array(transformed)

    scale = np.linalg.norm(transformed[9])

    if scale < 1e-9:
        return None

    transformed = transformed / scale

    return transformed


def geometry_features(pts):

    features = []

    def d(a, b):
        return np.linalg.norm(
            pts[a] - pts[b]
        )

    # Normalized landmarks
    for i in range(21):
        features.extend([
            pts[i, 0],
            pts[i, 1],
            pts[i, 2]
        ])

    # Finger geometry
    fingers = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20)
    ]

    for mcp, pip, dip, tip in fingers:

        features.extend([
            d(0, mcp),
            d(0, pip),
            d(0, dip),
            d(0, tip),
            d(mcp, pip),
            d(pip, dip),
            d(dip, tip),
            d(mcp, tip)
        ])

    # Fingertip distances
    tips = [4, 8, 12, 16, 20]

    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            features.append(
                d(tips[i], tips[j])
            )

    # Fingertip relationships
    for a, b in [
        (8, 12),
        (12, 16),
        (16, 20),
        (8, 16),
        (8, 20),
        (12, 20)
    ]:
        features.extend([
            pts[a, 0] - pts[b, 0],
            pts[a, 1] - pts[b, 1],
            pts[a, 2] - pts[b, 2]
        ])

    # Thumb relationships
    for tip in [8, 12, 16, 20]:
        features.append(
            d(4, tip)
        )

    return np.array(features)


print("=" * 75)
print("V15 ORIENTATION-NORMALIZED HAND REPRESENTATION")
print("=" * 75)

df = pd.read_csv(DATASET)

print(f"\nRows: {len(df)}")
print(f"Gestures: {df['label'].nunique()}")
print(f"Sessions: {df['session_id'].nunique()}")

print("\nGenerating orientation-normalized features...")

rows = []

for _, row in df.iterrows():

    pts = normalize_hand(row)

    if pts is None:
        continue

    feat = geometry_features(pts)

    item = {}

    for i, value in enumerate(feat):
        item[f"f_{i}"] = value

    item["label"] = row["label"]
    item["session_id"] = row["session_id"]

    rows.append(item)

v15 = pd.DataFrame(rows)

feature_cols = [
    c for c in v15.columns
    if c not in ["label", "session_id"]
]

print(f"\nValid samples: {len(v15)}")
print(f"Feature count: {len(feature_cols)}")

# ---------------------------------------------------------
# Session-independent evaluation
# ---------------------------------------------------------

train_mask = v15["session_id"].str.contains(
    "_S1_|_S2_"
)

test_mask = v15["session_id"].str.contains(
    "_S3_"
)

train = v15[train_mask].copy()
test = v15[test_mask].copy()

X_train = train[feature_cols]
y_train = train["label"]

X_test = test[feature_cols]
y_test = test["label"]

print(f"\nTraining samples: {len(train)}")
print(f"Unseen Session 3 samples: {len(test)}")

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced"
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results[name] = (
        accuracy,
        model,
        prediction
    )

    print(
        f"{name}: {accuracy * 100:.2f}%"
    )

# ---------------------------------------------------------
# Best model
# ---------------------------------------------------------

best_name = max(
    results,
    key=lambda name: results[name][0]
)

best_accuracy, best_model, best_prediction = results[
    best_name
]

print("\n" + "=" * 75)
print("V15 BEST RESULT")
print("=" * 75)

print(
    f"\nBest model: {best_name}"
)

print(
    f"Unseen-session accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print("\nClassification report:\n")

report = classification_report(
    y_test,
    best_prediction,
    labels=GESTURES,
    zero_division=0
)

print(report)

print("\nConfusion matrix:\n")

cm = confusion_matrix(
    y_test,
    best_prediction,
    labels=GESTURES
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_" + g
        for g in GESTURES
    ],
    columns=[
        "Pred_" + g
        for g in GESTURES
    ]
)

print(cm_df.to_string())

# ---------------------------------------------------------
# Per gesture accuracy
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PER-GESTURE ACCURACY")
print("=" * 75)

per_gesture = {}

for gesture in GESTURES:

    mask = y_test.values == gesture

    accuracy = (
        best_prediction[mask] == gesture
    ).mean()

    per_gesture[gesture] = accuracy

    print(
        f"{gesture:12s}: "
        f"{accuracy * 100:6.2f}%"
    )

# ---------------------------------------------------------
# Save feature dataset
# ---------------------------------------------------------

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "gesture_v15_orientation_features.csv"
)

v15.to_csv(
    FEATURE_FILE,
    index=False
)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V15 ORIENTATION-NORMALIZED HAND REPRESENTATION\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write(
        f"Valid samples: {len(v15)}\n"
    )

    f.write(
        f"Feature count: {len(feature_cols)}\n"
    )

    f.write(
        f"Training samples: {len(train)}\n"
    )

    f.write(
        f"Unseen Session 3: {len(test)}\n\n"
    )

    f.write(
        "MODEL COMPARISON\n"
    )

    for name, result in results.items():

        f.write(
            f"{name}: "
            f"{result[0] * 100:.2f}%\n"
        )

    f.write(
        f"\nBEST MODEL: {best_name}\n"
    )

    f.write(
        f"BEST ACCURACY: "
        f"{best_accuracy * 100:.2f}%\n\n"
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

    for gesture, accuracy in per_gesture.items():

        f.write(
            f"{gesture}: "
            f"{accuracy * 100:.2f}%\n"
        )

print("\n" + "=" * 75)
print("V15 COMPLETE")
print("=" * 75)

print(
    f"\nFeature dataset:\n{FEATURE_FILE}"
)

print(
    f"\nReport:\n{REPORT}"
)

print(
    "\nEvaluation was performed strictly on unseen Session 3."
)
