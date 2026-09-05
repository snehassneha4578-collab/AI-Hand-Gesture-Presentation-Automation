from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "dataset" / "gesture_dataset_v7.csv"
OUTPUT = BASE_DIR / "dataset" / "gesture_dataset_v8_features.csv"

print("=" * 75)
print("V8 ROBUST FEATURE ENGINEERING")
print("=" * 75)

df = pd.read_csv(INPUT)

LANDMARKS = 21

# ------------------------------------------------------------
# MediaPipe hand landmark indices
# ------------------------------------------------------------
WRIST = 0

THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

FINGERS = [
    THUMB,
    INDEX,
    MIDDLE,
    RING,
    PINKY
]

TIPS = [4, 8, 12, 16, 20]
MCPS = [2, 5, 9, 13, 17]


# ------------------------------------------------------------
# Load raw landmarks
# ------------------------------------------------------------

feature_cols = [
    f"{axis}_{i}"
    for i in range(21)
    for axis in ["x", "y", "z"]
]

raw = df[feature_cols].to_numpy(dtype=np.float64)

landmarks = raw.reshape(
    len(df),
    21,
    3
)


# ------------------------------------------------------------
# Vector helpers
# ------------------------------------------------------------

def safe_norm(v):
    value = np.linalg.norm(v)

    if value < 1e-8:
        return 1e-8

    return value


def unit(v):
    n = safe_norm(v)
    return v / n


def angle(a, b, c):

    v1 = a - b
    v2 = c - b

    n1 = safe_norm(v1)
    n2 = safe_norm(v2)

    cosine = np.dot(v1, v2) / (n1 * n2)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.arccos(cosine)


# ------------------------------------------------------------
# V8 feature extraction
# ------------------------------------------------------------

def extract_features(hand):

    wrist = hand[WRIST]

    # --------------------------------------------------------
    # 1. Wrist-centered coordinates
    # --------------------------------------------------------

    centered = hand - wrist

    # --------------------------------------------------------
    # 2. Scale normalization
    #
    # Use palm size rather than a single finger.
    # --------------------------------------------------------

    palm_vectors = np.array([
        hand[5] - hand[0],
        hand[9] - hand[0],
        hand[13] - hand[0],
        hand[17] - hand[0]
    ])

    palm_scale = np.mean(
        np.linalg.norm(
            palm_vectors,
            axis=1
        )
    )

    palm_scale = max(
        palm_scale,
        1e-6
    )

    normalized = centered / palm_scale

    features = []

    # --------------------------------------------------------
    # 3. Normalized landmark coordinates
    # --------------------------------------------------------

    features.extend(
        normalized.flatten()
    )

    # --------------------------------------------------------
    # 4. Finger segment lengths
    # --------------------------------------------------------

    for finger in FINGERS:

        for i in range(len(finger) - 1):

            a = hand[finger[i]]
            b = hand[finger[i + 1]]

            features.append(
                np.linalg.norm(a - b)
                / palm_scale
            )

    # --------------------------------------------------------
    # 5. Finger total path lengths
    # --------------------------------------------------------

    for finger in FINGERS:

        path = 0.0

        for i in range(len(finger) - 1):

            path += np.linalg.norm(
                hand[finger[i + 1]]
                -
                hand[finger[i]]
            )

        features.append(
            path / palm_scale
        )

    # --------------------------------------------------------
    # 6. Finger tip → wrist distances
    # --------------------------------------------------------

    for tip in TIPS:

        features.append(
            np.linalg.norm(
                hand[tip] - wrist
            )
            / palm_scale
        )

    # --------------------------------------------------------
    # 7. Finger tip pairwise distances
    # --------------------------------------------------------

    for i in range(len(TIPS)):

        for j in range(i + 1, len(TIPS)):

            features.append(
                np.linalg.norm(
                    hand[TIPS[i]]
                    -
                    hand[TIPS[j]]
                )
                / palm_scale
            )

    # --------------------------------------------------------
    # 8. Joint bending angles
    # --------------------------------------------------------

    angle_triplets = [

        # Thumb
        (1, 2, 3),
        (2, 3, 4),

        # Index
        (5, 6, 7),
        (6, 7, 8),

        # Middle
        (9, 10, 11),
        (10, 11, 12),

        # Ring
        (13, 14, 15),
        (14, 15, 16),

        # Pinky
        (17, 18, 19),
        (18, 19, 20)
    ]

    for a, b, c in angle_triplets:

        features.append(
            angle(
                hand[a],
                hand[b],
                hand[c]
            )
        )

    # --------------------------------------------------------
    # 9. Finger extension / curl ratios
    # --------------------------------------------------------

    for finger in FINGERS:

        tip = hand[finger[-1]]
        base = hand[finger[0]]

        path = 0.0

        for i in range(len(finger) - 1):

            path += np.linalg.norm(
                hand[finger[i + 1]]
                -
                hand[finger[i]]
            )

        chord = np.linalg.norm(
            tip - base
        )

        features.append(
            chord / max(path, 1e-6)
        )

    # --------------------------------------------------------
    # 10. Tip height relationships
    # --------------------------------------------------------

    for i in range(len(TIPS)):

        for j in range(i + 1, len(TIPS)):

            features.append(
                (
                    normalized[TIPS[i], 1]
                    -
                    normalized[TIPS[j], 1]
                )
            )

    # --------------------------------------------------------
    # 11. Tip horizontal relationships
    # --------------------------------------------------------

    for i in range(len(TIPS)):

        for j in range(i + 1, len(TIPS)):

            features.append(
                (
                    normalized[TIPS[i], 0]
                    -
                    normalized[TIPS[j], 0]
                )
            )

    # --------------------------------------------------------
    # 12. Palm geometry
    # --------------------------------------------------------

    palm_pairs = [
        (5, 17),
        (9, 13),
        (5, 13),
        (9, 17)
    ]

    for a, b in palm_pairs:

        features.append(
            np.linalg.norm(
                hand[a] - hand[b]
            )
            / palm_scale
        )

    # --------------------------------------------------------
    # 13. Thumb-index relationship
    # --------------------------------------------------------

    features.append(
        np.linalg.norm(
            hand[4] - hand[8]
        )
        / palm_scale
    )

    # Thumb-middle
    features.append(
        np.linalg.norm(
            hand[4] - hand[12]
        )
        / palm_scale
    )

    # Thumb-pinky
    features.append(
        np.linalg.norm(
            hand[4] - hand[20]
        )
        / palm_scale
    )

    # --------------------------------------------------------
    # 14. Finger-to-finger opening
    # --------------------------------------------------------

    for i in range(4):

        features.append(
            np.linalg.norm(
                hand[TIPS[i]]
                -
                hand[TIPS[i + 1]]
            )
            / palm_scale
        )

    return np.asarray(
        features,
        dtype=np.float64
    )


# ------------------------------------------------------------
# Build V8 matrix
# ------------------------------------------------------------

print("\nExtracting V8 features...")

feature_matrix = []

for i, hand in enumerate(landmarks):

    feature_matrix.append(
        extract_features(hand)
    )

X = np.asarray(
    feature_matrix,
    dtype=np.float64
)

print(
    f"V8 feature matrix: {X.shape}"
)

print(
    f"Feature count: {X.shape[1]}"
)


# ------------------------------------------------------------
# Check numerical validity
# ------------------------------------------------------------

finite_mask = np.isfinite(X).all(axis=1)

print(
    f"Valid rows: {finite_mask.sum()}/{len(X)}"
)

if not finite_mask.all():

    print(
        "WARNING: Removing non-finite rows."
    )

    X = X[finite_mask]
    df = df.iloc[
        np.where(finite_mask)[0]
    ].copy()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

v8_columns = [
    f"f_{i}"
    for i in range(X.shape[1])
]

output_df = pd.DataFrame(
    X,
    columns=v8_columns
)

output_df["label"] = df["label"].values
output_df["session_id"] = df["session_id"].values

output_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 75)
print("V8 FEATURE ENGINEERING COMPLETE")
print("=" * 75)

print(
    f"Rows: {len(output_df)}"
)

print(
    f"Features: {len(v8_columns)}"
)

print(
    f"Sessions: "
    f"{output_df['session_id'].nunique()}"
)

print("\nGESTURE COUNTS")

print(
    output_df["label"]
    .value_counts()
    .sort_index()
)

print(
    f"\nSaved to:\n{OUTPUT}"
)
