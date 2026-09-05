import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")
REPORT = os.path.join(BASE_DIR, "dataset", "v14_hierarchical_results.txt")

df = pd.read_csv(DATASET)

GESTURES = ["fist", "one", "open_palm", "pinch", "three", "two"]

# ---------------------------------------------------------
# Landmark utilities
# ---------------------------------------------------------

def p(row, i):
    return np.array([
        float(row[f"x_{i}"]),
        float(row[f"y_{i}"]),
        float(row[f"z_{i}"])
    ])

def dist(a, b):
    return np.linalg.norm(a - b)

# ---------------------------------------------------------
# Geometry features
# ---------------------------------------------------------

def extract(row):

    w = p(row, 0)

    fingers = {
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20)
    }

    middle_mcp = p(row, 9)
    scale = dist(w, middle_mcp) + 1e-9

    out = {}

    # Finger extension / geometry
    for name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():

        mcp = p(row, mcp_i)
        pip = p(row, pip_i)
        dip = p(row, dip_i)
        tip = p(row, tip_i)

        out[f"{name}_ratio"] = (
            dist(w, tip) / (dist(w, mcp) + 1e-9)
        )

        out[f"{name}_tip_wrist"] = (
            dist(w, tip) / scale
        )

        out[f"{name}_tip_mcp"] = (
            dist(tip, mcp) / scale
        )

        out[f"{name}_pip_wrist"] = (
            dist(w, pip) / scale
        )

    tips = {
        "index": p(row, 8),
        "middle": p(row, 12),
        "ring": p(row, 16),
        "pinky": p(row, 20),
        "thumb": p(row, 4)
    }

    # Stable fingertip distances
    for a, b in [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
        ("index", "ring"),
        ("index", "pinky"),
        ("middle", "pinky"),
        ("thumb", "index"),
        ("thumb", "middle")
    ]:
        out[f"{a}_{b}"] = (
            dist(tips[a], tips[b]) / scale
        )

    # Stable fingertip relationships
    out["index_middle_y"] = (
        tips["index"][1] - tips["middle"][1]
    ) / scale

    out["middle_ring_y"] = (
        tips["middle"][1] - tips["ring"][1]
    ) / scale

    out["ring_pinky_y"] = (
        tips["ring"][1] - tips["pinky"][1]
    ) / scale

    out["index_middle_x"] = (
        tips["index"][0] - tips["middle"][0]
    ) / scale

    out["middle_ring_x"] = (
        tips["middle"][0] - tips["ring"][0]
    ) / scale

    out["ring_pinky_x"] = (
        tips["ring"][0] - tips["pinky"][0]
    ) / scale

    return out


print("=" * 75)
print("V14 HIERARCHICAL DATA-DRIVEN GESTURE RECOGNIZER")
print("=" * 75)

# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

print("\nExtracting geometric features...")

rows = []

for _, row in df.iterrows():

    features = extract(row)
    features["label"] = row["label"]
    features["session_id"] = row["session_id"]

    rows.append(features)

g = pd.DataFrame(rows)

feature_columns = [
    c for c in g.columns
    if c not in ["label", "session_id"]
]

# ---------------------------------------------------------
# Identify sessions
# ---------------------------------------------------------

session_order = (
    g.groupby(["label", "session_id"])
    .size()
    .reset_index()
)

session_ids = list(
    g["session_id"].unique()
)

# The dataset has S1/S2/S3.
# Use S1 + S2 for training and S3 for unseen testing.

train_mask = g["session_id"].str.contains("_S1_|_S2_")
test_mask = g["session_id"].str.contains("_S3_")

train = g[train_mask].copy()
test = g[test_mask].copy()

print(f"\nTraining samples: {len(train)}")
print(f"Unseen test samples: {len(test)}")

print("\nTraining distribution:")
print(train["label"].value_counts().sort_index())

print("\nTest distribution:")
print(test["label"].value_counts().sort_index())

# ---------------------------------------------------------
# Select stable / discriminative features
# ---------------------------------------------------------

SELECTED = [
    "ring_pinky_y",
    "middle_ring_y",
    "index_middle_y",

    "pinky_ratio",
    "ring_ratio",
    "middle_ratio",
    "index_ratio",

    "pinky_tip_wrist",
    "ring_tip_wrist",
    "middle_tip_wrist",
    "index_tip_wrist",

    "pinky_pip_wrist",
    "ring_pip_wrist",
    "middle_pip_wrist",
    "index_pip_wrist",

    "index_middle",
    "middle_ring",
    "ring_pinky",
    "index_ring",
    "index_pinky",
    "middle_pinky",

    "thumb_index",
    "thumb_middle"
]

SELECTED = [
    f for f in SELECTED
    if f in feature_columns
]

print("\nSelected geometric features:")
for f in SELECTED:
    print(" ", f)

# ---------------------------------------------------------
# Hierarchical classifiers
# ---------------------------------------------------------
#
# Stage 1:
# Pinch vs non-pinch
#
# Stage 2:
# Open palm vs non-open-palm
#
# Stage 3:
# Closed-family classifier:
# fist / one / two / three
#
# ---------------------------------------------------------

def train_binary(X, y):

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "tree",
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    model.fit(X, y)

    return model


X_train = train[SELECTED]
X_test = test[SELECTED]

# Stage 1
y_pinch_train = (
    train["label"] == "pinch"
).astype(int)

pinch_model = train_binary(
    X_train,
    y_pinch_train
)

pinch_pred = pinch_model.predict(X_test)

# Stage 2
non_pinch_train = train[
    train["label"] != "pinch"
]

open_X = non_pinch_train[SELECTED]

open_y = (
    non_pinch_train["label"] == "open_palm"
).astype(int)

open_model = train_binary(
    open_X,
    open_y
)

open_pred = open_model.predict(
    X_test
)

# Stage 3
closed_train = train[
    ~train["label"].isin(
        ["pinch", "open_palm"]
    )
]

closed_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "tree",
        DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42
        )
    )
])

closed_model.fit(
    closed_train[SELECTED],
    closed_train["label"]
)

closed_pred = closed_model.predict(
    X_test
)

# ---------------------------------------------------------
# Combine hierarchy
# ---------------------------------------------------------

final_pred = []

for i in range(len(test)):

    if pinch_pred[i] == 1:

        prediction = "pinch"

    elif open_pred[i] == 1:

        prediction = "open_palm"

    else:

        prediction = closed_pred[i]

    final_pred.append(prediction)

final_pred = np.array(final_pred)

# ---------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(
    test["label"],
    final_pred
)

print("\n" + "=" * 75)
print("V14 UNSEEN-SESSION RESULT")
print("=" * 75)

print(
    f"\nOverall accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification report:")

print(
    classification_report(
        test["label"],
        final_pred,
        labels=GESTURES,
        zero_division=0
    )
)

print("\nConfusion matrix:")

cm = confusion_matrix(
    test["label"],
    final_pred,
    labels=GESTURES
)

print(
    pd.DataFrame(
        cm,
        index=[
            f"Actual_{x}"
            for x in GESTURES
        ],
        columns=[
            f"Pred_{x}"
            for x in GESTURES
        ]
    )
)

# ---------------------------------------------------------
# Per gesture accuracy
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PER-GESTURE ACCURACY")
print("=" * 75)

per_gesture = {}

for gesture in GESTURES:

    mask = test["label"].values == gesture

    if mask.sum() == 0:
        continue

    acc = (
        final_pred[mask] == gesture
    ).mean()

    per_gesture[gesture] = acc

    print(
        f"{gesture:12s}: {acc * 100:6.2f}%"
    )

# ---------------------------------------------------------
# Stage accuracy
# ---------------------------------------------------------

actual_pinch = (
    test["label"] == "pinch"
).astype(int)

actual_open = (
    test["label"] == "open_palm"
).astype(int)

pinch_stage_acc = accuracy_score(
    actual_pinch,
    pinch_pred
)

open_stage_acc = accuracy_score(
    actual_open,
    open_pred
)

closed_mask = ~test["label"].isin(
    ["pinch", "open_palm"]
)

closed_acc = accuracy_score(
    test.loc[closed_mask, "label"],
    closed_pred[closed_mask.values]
)

print("\n" + "=" * 75)
print("HIERARCHY STAGE ACCURACY")
print("=" * 75)

print(
    f"Pinch detection:     {pinch_stage_acc * 100:.2f}%"
)

print(
    f"Open-palm detection: {open_stage_acc * 100:.2f}%"
)

print(
    f"Closed gestures:     {closed_acc * 100:.2f}%"
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
        "V14 HIERARCHICAL DATA-DRIVEN GESTURE RECOGNIZER\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write(
        f"Training samples: {len(train)}\n"
    )

    f.write(
        f"Unseen test samples: {len(test)}\n"
    )

    f.write(
        f"Overall accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "SELECTED FEATURES\n"
    )

    for feature in SELECTED:
        f.write(
            f"{feature}\n"
        )

    f.write(
        "\nCLASSIFICATION REPORT\n"
    )

    f.write(
        classification_report(
            test["label"],
            final_pred,
            labels=GESTURES,
            zero_division=0
        )
    )

    f.write(
        "\nCONFUSION MATRIX\n"
    )

    f.write(
        pd.DataFrame(
            cm,
            index=[
                f"Actual_{x}"
                for x in GESTURES
            ],
            columns=[
                f"Pred_{x}"
                for x in GESTURES
            ]
        ).to_string()
    )

    f.write(
        "\n\nPER-GESTURE ACCURACY\n"
    )

    for gesture, acc in per_gesture.items():

        f.write(
            f"{gesture}: {acc * 100:.2f}%\n"
        )

    f.write(
        "\nHIERARCHY STAGE ACCURACY\n"
    )

    f.write(
        f"Pinch: {pinch_stage_acc * 100:.2f}%\n"
    )

    f.write(
        f"Open palm: {open_stage_acc * 100:.2f}%\n"
    )

    f.write(
        f"Closed: {closed_acc * 100:.2f}%\n"
    )

print("\n" + "=" * 75)
print("V14 COMPLETE")
print("=" * 75)

print(
    f"\nReport saved to:\n{REPORT}"
)

print(
    "\nIMPORTANT: This was evaluated on completely unseen Session 3."
)
