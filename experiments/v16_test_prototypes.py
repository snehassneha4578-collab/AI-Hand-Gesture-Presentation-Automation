import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

# Correct the path automatically if the folder is actually one OneDrive
if not os.path.exists(BASE_DIR):
    BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_calibration",
    "v16_personal_calibration.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "trained_model",
    "v16_personal_prototypes.pkl"
)

REPORT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_prototype_test_results.txt"
)

GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]


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

    # 63 normalized landmark values
    for point in points:
        features.extend(point.tolist())

    # Wrist-to-fingertip distances
    for index in [4, 8, 12, 16, 20]:

        features.append(
            np.linalg.norm(
                points[index] - points[0]
            )
        )

    # Pairwise fingertip distances
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
print("V16 PERSONALIZED PROTOTYPE TEST")
print("=" * 70)

if not os.path.exists(DATASET):

    print("\nERROR: Dataset not found:")
    print(DATASET)
    raise SystemExit

if not os.path.exists(MODEL_FILE):

    print("\nERROR: Prototype model not found:")
    print(MODEL_FILE)
    raise SystemExit


df = pd.read_csv(DATASET)

with open(
    MODEL_FILE,
    "rb"
) as f:

    model = pickle.load(f)


prototypes = model["prototypes"]

print(
    f"\nSamples: {len(df)}"
)

print(
    f"Feature dimension: "
    f"{model['feature_dimension']}"
)


# ---------------------------------------------------------
# Generate features
# ---------------------------------------------------------

X = []
y = []

for _, row in df.iterrows():

    vector = build_feature_vector(row)

    if vector is None:
        continue

    X.append(vector)
    y.append(row["label"])


X = np.array(X)
y = np.array(y)


print(
    f"Valid samples: {len(X)}"
)


# ---------------------------------------------------------
# Prototype prediction
# ---------------------------------------------------------

predictions = []
nearest_distances = []

for vector in X:

    distances = {}

    for gesture in GESTURES:

        prototype = prototypes[gesture]["prototype"]

        distances[gesture] = np.linalg.norm(
            vector - prototype
        )

    best_gesture = min(
        distances,
        key=distances.get
    )

    predictions.append(
        best_gesture
    )

    nearest_distances.append(
        distances[best_gesture]
    )


predictions = np.array(predictions)


# ---------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------

accuracy = np.mean(
    predictions == y
)


print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    f"\nPrototype recognition accuracy: "
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
print("PER-GESTURE ACCURACY")
print("=" * 70)

per_gesture = {}

for gesture in GESTURES:

    mask = y == gesture

    gesture_accuracy = np.mean(
        predictions[mask] == gesture
    )

    per_gesture[gesture] = gesture_accuracy

    print(
        f"{gesture:12s}: "
        f"{gesture_accuracy * 100:6.2f}%"
    )


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

matrix = np.zeros(
    (len(GESTURES), len(GESTURES)),
    dtype=int
)

for actual, predicted in zip(
    y,
    predictions
):

    actual_index = GESTURES.index(actual)
    predicted_index = GESTURES.index(predicted)

    matrix[
        actual_index,
        predicted_index
    ] += 1


cm_df = pd.DataFrame(
    matrix,
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
print("DISTANCE STATISTICS")
print("=" * 70)

print(
    f"\nMean nearest-prototype distance: "
    f"{np.mean(nearest_distances):.4f}"
)

print(
    f"Median nearest-prototype distance: "
    f"{np.median(nearest_distances):.4f}"
)

print(
    f"Maximum nearest-prototype distance: "
    f"{np.max(nearest_distances):.4f}"
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
        "V16 PERSONALIZED PROTOTYPE TEST\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Samples tested: {len(X)}\n"
    )

    f.write(
        f"Feature dimension: "
        f"{model['feature_dimension']}\n"
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

    f.write(
        "\n\nDISTANCE STATISTICS\n"
    )

    f.write(
        f"Mean: {np.mean(nearest_distances):.4f}\n"
    )

    f.write(
        f"Median: {np.median(nearest_distances):.4f}\n"
    )

    f.write(
        f"Maximum: {np.max(nearest_distances):.4f}\n"
    )


print("\n" + "=" * 70)
print("V16 STEP 3 COMPLETE")
print("=" * 70)

print(
    f"\nReport saved to:\n{REPORT_FILE}"
)
