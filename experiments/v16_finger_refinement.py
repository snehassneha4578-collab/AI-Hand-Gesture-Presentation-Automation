import pandas as pd
import numpy as np
import os
import pickle

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_calibration",
    "v16_personal_calibration.csv"
)

REPORT = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_finger_refinement_results.txt"
)

GESTURES = [
    "one",
    "two",
    "three"
]


def get_points(row):

    return np.array([
        [
            row[f"x_{i}"],
            row[f"y_{i}"],
            row[f"z_{i}"]
        ]
        for i in range(21)
    ], dtype=float)


def normalize(points):

    # Wrist origin
    points = points - points[0]

    # Scale using wrist -> middle MCP
    scale = np.linalg.norm(points[9])

    if scale < 1e-9:
        return None

    points = points / scale

    return points


def build_finger_features(row):

    points = normalize(
        get_points(row)
    )

    if points is None:
        return None

    features = []

    # Fingertip coordinates
    tips = [8, 12, 16, 20]

    for tip in tips:

        features.extend([
            points[tip, 0],
            points[tip, 1],
            points[tip, 2]
        ])

    # Fingertip -> wrist distances
    for tip in tips:

        features.append(
            np.linalg.norm(
                points[tip] - points[0]
            )
        )

    # Finger segment ratios
    fingers = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20)
    ]

    for mcp, pip, dip, tip in fingers:

        mcp_wrist = np.linalg.norm(
            points[mcp] - points[0]
        )

        tip_wrist = np.linalg.norm(
            points[tip] - points[0]
        )

        ratio = (
            tip_wrist /
            (mcp_wrist + 1e-9)
        )

        features.append(ratio)

    # Adjacent fingertip distances
    for a, b in [
        (8, 12),
        (12, 16),
        (16, 20)
    ]:

        features.append(
            np.linalg.norm(
                points[a] - points[b]
            )
        )

    # Important vertical relationships
    features.extend([
        points[8, 1] - points[12, 1],
        points[12, 1] - points[16, 1],
        points[16, 1] - points[20, 1],
        points[8, 1] - points[16, 1],
        points[12, 1] - points[20, 1]
    ])

    # Important horizontal relationships
    features.extend([
        points[8, 0] - points[12, 0],
        points[12, 0] - points[16, 0],
        points[16, 0] - points[20, 0]
    ])

    return np.array(
        features,
        dtype=float
    )


print("=" * 70)
print("V16 STEP 5 - FINGER COUNT REFINEMENT")
print("=" * 70)

df = pd.read_csv(DATASET)

print(
    f"\nTotal samples: {len(df)}"
)

# ---------------------------------------------------------
# 70/30 split
# ---------------------------------------------------------

prototype_rows = []
validation_rows = []

for gesture in GESTURES:

    gesture_df = df[
        df["label"] == gesture
    ].reset_index(drop=True)

    prototype_rows.append(
        gesture_df.iloc[:70]
    )

    validation_rows.append(
        gesture_df.iloc[70:]
    )

prototype_df = pd.concat(
    prototype_rows,
    ignore_index=True
)

validation_df = pd.concat(
    validation_rows,
    ignore_index=True
)

print(
    f"\nPrototype samples: "
    f"{len(prototype_df)}"
)

print(
    f"Validation samples: "
    f"{len(validation_df)}"
)

# ---------------------------------------------------------
# Build specialized prototypes
# ---------------------------------------------------------

prototypes = {}

for gesture in GESTURES:

    gesture_df = prototype_df[
        prototype_df["label"] == gesture
    ]

    vectors = []

    for _, row in gesture_df.iterrows():

        vector = build_finger_features(row)

        if vector is not None:
            vectors.append(vector)

    X = np.array(vectors)

    prototypes[gesture] = np.mean(
        X,
        axis=0
    )

    print(
        f"{gesture:8s}: "
        f"{len(X)} samples"
    )


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

y_true = []
y_pred = []

for _, row in validation_df.iterrows():

    vector = build_finger_features(row)

    if vector is None:
        continue

    distances = {}

    for gesture in GESTURES:

        distances[gesture] = np.linalg.norm(
            vector - prototypes[gesture]
        )

    prediction = min(
        distances,
        key=distances.get
    )

    y_true.append(
        row["label"]
    )

    y_pred.append(
        prediction
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

accuracy = np.mean(
    y_true == y_pred
)

print("\n" + "=" * 70)
print("FINGER COUNT RESULT")
print("=" * 70)

print(
    f"\nOne/Two/Three accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "\nRandom three-class baseline: "
    "33.33%"
)


print("\nPER-GESTURE ACCURACY")

per_gesture = {}

for gesture in GESTURES:

    mask = y_true == gesture

    score = np.mean(
        y_pred[mask] == gesture
    )

    per_gesture[gesture] = score

    print(
        f"{gesture:8s}: "
        f"{score * 100:6.2f}%"
    )


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = np.zeros(
    (3, 3),
    dtype=int
)

for actual, predicted in zip(
    y_true,
    y_pred
):

    i = GESTURES.index(actual)
    j = GESTURES.index(predicted)

    cm[i, j] += 1


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

print("\nCONFUSION MATRIX")

print(
    cm_df.to_string()
)


# ---------------------------------------------------------
# Save specialized model
# ---------------------------------------------------------

MODEL_FILE = os.path.join(
    BASE_DIR,
    "trained_model",
    "v16_finger_refinement_prototypes.pkl"
)

with open(
    MODEL_FILE,
    "wb"
) as f:

    pickle.dump(
        {
            "version": "V16",
            "gestures": GESTURES,
            "prototypes": prototypes
        },
        f
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
        "V16 FINGER COUNT REFINEMENT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Prototype samples: "
        f"{len(prototype_df)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(validation_df)}\n"
    )

    f.write(
        f"Overall accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "PER-GESTURE ACCURACY\n"
    )

    for gesture in GESTURES:

        f.write(
            f"{gesture}: "
            f"{per_gesture[gesture] * 100:.2f}%\n"
        )

    f.write(
        "\nCONFUSION MATRIX\n"
    )

    f.write(
        cm_df.to_string()
    )


print("\n" + "=" * 70)
print("V16 STEP 5 COMPLETE")
print("=" * 70)

print(
    f"\nSpecialized model saved to:\n{MODEL_FILE}"
)

print(
    f"\nReport saved to:\n{REPORT}"
)
