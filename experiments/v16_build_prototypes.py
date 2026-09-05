import pandas as pd
import numpy as np
import os
import pickle

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_calibration",
    "v16_personal_calibration.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "trained_model"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "v16_personal_prototypes.pkl"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # Translation invariance
    points = points - points[0]

    # Scale reference: wrist -> middle MCP
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

    # Distances from wrist
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

    # Important finger relationships
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

    # Thumb relationships
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
print("V16 PERSONALIZED PROTOTYPE BUILDER")
print("=" * 70)

if not os.path.exists(INPUT_FILE):

    print("\nERROR: Calibration dataset not found:")
    print(INPUT_FILE)
    raise SystemExit


df = pd.read_csv(INPUT_FILE)

print(
    f"\nLoaded samples: {len(df)}"
)

print(
    f"Gestures: {df['label'].nunique()}"
)


prototype_data = {}

valid_counts = {}

for gesture in GESTURES:

    gesture_df = df[
        df["label"] == gesture
    ]

    feature_vectors = []

    for _, row in gesture_df.iterrows():

        vector = build_feature_vector(row)

        if vector is not None:
            feature_vectors.append(vector)

    X = np.array(
        feature_vectors,
        dtype=float
    )

    prototype = np.mean(
        X,
        axis=0
    )

    std = np.std(
        X,
        axis=0
    )

    # Average distance of calibration samples
    # from their gesture prototype
    distances = np.linalg.norm(
        X - prototype,
        axis=1
    )

    mean_distance = float(
        np.mean(distances)
    )

    std_distance = float(
        np.std(distances)
    )

    # Robust recognition radius.
    # We use mean + 2 standard deviations.
    recognition_radius = (
        mean_distance +
        2.0 * std_distance
    )

    prototype_data[gesture] = {
        "prototype": prototype,
        "feature_std": std,
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "recognition_radius": recognition_radius,
        "sample_count": len(X)
    }

    valid_counts[gesture] = len(X)

    print(
        f"\n{gesture.upper()}"
    )

    print(
        f"Samples: {len(X)}"
    )

    print(
        f"Feature dimensions: {X.shape[1]}"
    )

    print(
        f"Mean prototype distance: "
        f"{mean_distance:.4f}"
    )

    print(
        f"Recognition radius: "
        f"{recognition_radius:.4f}"
    )


model = {
    "version": "V16",
    "method": "Personalized Prototype Recognition",
    "gestures": GESTURES,
    "feature_dimension": len(
        next(iter(prototype_data.values()))["prototype"]
    ),
    "prototypes": prototype_data
}


with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print("\n" + "=" * 70)
print("V16 PROTOTYPE BUILD COMPLETE")
print("=" * 70)

print(
    f"\nFeature dimension: "
    f"{model['feature_dimension']}"
)

print("\nPrototype samples:")

for gesture in GESTURES:

    print(
        f"{gesture:12s}: "
        f"{valid_counts[gesture]}"
    )

print(
    f"\nSaved model to:\n{OUTPUT_FILE}"
)

print(
    "\nThese prototypes represent your "
    "personal hand geometry."
)
