import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_calibration",
    "v16_personal_calibration.csv"
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_generalization_results.txt"
)

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]

PROTOTYPE_SAMPLES = 70


def normalize_sample(row):

    points = np.array([
        [
            row[f"x_{i}"],
            row[f"y_{i}"],
            row[f"z_{i}"]
        ]
        for i in range(21)
    ], dtype=float)

    points = points - points[0]

    scale = np.linalg.norm(points[9])

    if scale < 1e-9:
        return None

    points = points / scale

    return points


def build_feature_vector(row):

    points = normalize_sample(row)

    if points is None:
        return None

    features = []

    # Normalized landmarks
    for point in points:
        features.extend(point.tolist())

    # Wrist-to-fingertip distances
    for index in [4, 8, 12, 16, 20]:
        features.append(
            np.linalg.norm(points[index] - points[0])
        )

    # Fingertip pair distances
    tips = [4, 8, 12, 16, 20]

    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            features.append(
                np.linalg.norm(
                    points[tips[i]] - points[tips[j]]
                )
            )

    # Relative fingertip positions
    pairs = [
        (8, 12),
        (12, 16),
        (16, 20),
        (8, 16),
        (8, 20),
        (12, 20)
    ]

    for a, b in pairs:

        difference = points[a] - points[b]

        features.extend(
            difference.tolist()
        )

    # Thumb-to-fingertip distances
    for tip in [8, 12, 16, 20]:

        features.append(
            np.linalg.norm(
                points[4] - points[tip]
            )
        )

    return np.array(
        features,
        dtype=float
    )


print("=" * 70)
print("V16 STEP 4 - PERSONALIZED GENERALIZATION TEST")
print("=" * 70)

if not os.path.exists(DATASET):

    print("\nERROR: Dataset not found:")
    print(DATASET)
    raise SystemExit


df = pd.read_csv(DATASET)

print(
    f"\nTotal samples: {len(df)}"
)


# ---------------------------------------------------------
# Create train/validation split
# ---------------------------------------------------------

prototype_rows = []
validation_rows = []

for gesture in GESTURES:

    gesture_df = df[
        df["label"] == gesture
    ].reset_index(drop=True)

    prototype_part = gesture_df.iloc[
        :PROTOTYPE_SAMPLES
    ]

    validation_part = gesture_df.iloc[
        PROTOTYPE_SAMPLES:
    ]

    prototype_rows.append(
        prototype_part
    )

    validation_rows.append(
        validation_part
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
    f"Unseen validation samples: "
    f"{len(validation_df)}"
)


# ---------------------------------------------------------
# Build prototypes ONLY from first 70 samples
# ---------------------------------------------------------

prototypes = {}

for gesture in GESTURES:

    gesture_df = prototype_df[
        prototype_df["label"] == gesture
    ]

    vectors = []

    for _, row in gesture_df.iterrows():

        vector = build_feature_vector(row)

        if vector is not None:
            vectors.append(vector)

    X = np.array(vectors)

    prototype = np.mean(
        X,
        axis=0
    )

    prototypes[gesture] = prototype


# ---------------------------------------------------------
# Test unseen 30 samples
# ---------------------------------------------------------

y_true = []
y_pred = []
distances = []


for _, row in validation_df.iterrows():

    vector = build_feature_vector(row)

    if vector is None:
        continue

    gesture_distances = {}

    for gesture in GESTURES:

        gesture_distances[gesture] = np.linalg.norm(
            vector - prototypes[gesture]
        )

    prediction = min(
        gesture_distances,
        key=gesture_distances.get
    )

    y_true.append(
        row["label"]
    )

    y_pred.append(
        prediction
    )

    distances.append(
        gesture_distances[prediction]
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ---------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------

accuracy = np.mean(
    y_true == y_pred
)


print("\n" + "=" * 70)
print("GENERALIZATION RESULT")
print("=" * 70)

print(
    f"\nUnseen validation accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "\nRandom six-class baseline: "
    "16.67%"
)


# ---------------------------------------------------------
# Per gesture
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PER-GESTURE VALIDATION ACCURACY")
print("=" * 70)

per_gesture = {}

for gesture in GESTURES:

    mask = y_true == gesture

    score = np.mean(
        y_pred[mask] == gesture
    )

    per_gesture[gesture] = score

    print(
        f"{gesture:12s}: "
        f"{score * 100:6.2f}%"
    )


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = np.zeros(
    (len(GESTURES), len(GESTURES)),
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

print(
    cm_df.to_string()
)


# ---------------------------------------------------------
# Distance statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("UNSEEN DISTANCE STATISTICS")
print("=" * 70)

print(
    f"\nMean distance: "
    f"{np.mean(distances):.4f}"
)

print(
    f"Median distance: "
    f"{np.median(distances):.4f}"
)

print(
    f"Maximum distance: "
    f"{np.max(distances):.4f}"
)


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V16 PERSONALIZED GENERALIZATION TEST\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Prototype samples: {len(prototype_df)}\n"
    )

    f.write(
        f"Validation samples: {len(validation_df)}\n"
    )

    f.write(
        f"Validation accuracy: "
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

    f.write(
        "\n\nDISTANCE STATISTICS\n"
    )

    f.write(
        f"Mean: {np.mean(distances):.4f}\n"
    )

    f.write(
        f"Median: {np.median(distances):.4f}\n"
    )

    f.write(
        f"Maximum: {np.max(distances):.4f}\n"
    )


print("\n" + "=" * 70)
print("V16 STEP 4 COMPLETE")
print("=" * 70)

print(
    f"\nReport saved to:\n{REPORT_FILE}"
)
