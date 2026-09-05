import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/gesture_dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def create_engineered_features(df):
    """
    Create geometric features from 21 MediaPipe hand landmarks.

    Each landmark contains:
        x, y, z

    Features created:
        1. Wrist-relative coordinates
        2. Scale-normalized coordinates
        3. Important landmark distances
        4. Finger extension measurements
    """

    # --------------------------------------------------------
    # Extract 21 landmarks
    # --------------------------------------------------------

    landmarks = []

    for i in range(21):

        x = df[f"x_{i}"].values
        y = df[f"y_{i}"].values
        z = df[f"z_{i}"].values

        landmarks.append(
            np.column_stack((x, y, z))
        )

    # Convert to:
    # (21, samples, 3)

    landmarks = np.array(landmarks)

    # Convert to:
    # (samples, 21, 3)

    landmarks = np.transpose(
        landmarks,
        (1, 0, 2)
    )

    # --------------------------------------------------------
    # 1. Wrist-relative coordinates
    # --------------------------------------------------------

    wrist = landmarks[:, 0, :]

    relative = (
        landmarks
        - wrist[:, np.newaxis, :]
    )

    relative_features = relative.reshape(
        len(df),
        -1
    )

    relative_columns = []

    for i in range(21):

        relative_columns.extend([
            f"rel_x_{i}",
            f"rel_y_{i}",
            f"rel_z_{i}"
        ])

    # --------------------------------------------------------
    # 2. Calculate hand scale
    # --------------------------------------------------------

    # Wrist = landmark 0
    # Middle finger MCP = landmark 9

    hand_scale = np.linalg.norm(
        landmarks[:, 9, :]
        - landmarks[:, 0, :],
        axis=1
    )

    # Prevent division by zero

    hand_scale = np.where(
        hand_scale < 1e-6,
        1e-6,
        hand_scale
    )

    # --------------------------------------------------------
    # 3. Scale-normalized coordinates
    # --------------------------------------------------------

    normalized_relative = (
        relative
        / hand_scale[:, np.newaxis, np.newaxis]
    )

    normalized_features = (
        normalized_relative.reshape(
            len(df),
            -1
        )
    )

    normalized_columns = []

    for i in range(21):

        normalized_columns.extend([
            f"norm_x_{i}",
            f"norm_y_{i}",
            f"norm_z_{i}"
        ])

    # --------------------------------------------------------
    # 4. Important landmark distances
    # --------------------------------------------------------

    distance_pairs = [

        # Wrist to fingertips
        (0, 4),
        (0, 8),
        (0, 12),
        (0, 16),
        (0, 20),

        # Thumb relationships
        (4, 8),
        (4, 12),

        # Fingertip relationships
        (8, 12),
        (12, 16),
        (16, 20),
        (8, 20)
    ]

    distance_features = []
    distance_columns = []

    for a, b in distance_pairs:

        distance = np.linalg.norm(
            landmarks[:, a, :]
            - landmarks[:, b, :],
            axis=1
        )

        # Make distance independent
        # of hand size

        normalized_distance = (
            distance / hand_scale
        )

        distance_features.append(
            normalized_distance
        )

        distance_columns.append(
            f"distance_{a}_{b}"
        )

    distance_features = np.column_stack(
        distance_features
    )

    # --------------------------------------------------------
    # 5. Finger extension features
    # --------------------------------------------------------

    finger_pairs = [
        (0, 4),
        (0, 8),
        (0, 12),
        (0, 16),
        (0, 20)
    ]

    finger_features = []
    finger_columns = []

    for a, b in finger_pairs:

        distance = np.linalg.norm(
            landmarks[:, a, :]
            - landmarks[:, b, :],
            axis=1
        )

        normalized_distance = (
            distance / hand_scale
        )

        finger_features.append(
            normalized_distance
        )

        finger_columns.append(
            f"finger_extension_{b}"
        )

    finger_features = np.column_stack(
        finger_features
    )

    # --------------------------------------------------------
    # 6. Combine all features
    # --------------------------------------------------------

    engineered = np.column_stack([
        relative_features,
        normalized_features,
        distance_features,
        finger_features
    ])

    all_columns = (
        relative_columns
        + normalized_columns
        + distance_columns
        + finger_columns
    )

    engineered_df = pd.DataFrame(
        engineered,
        columns=all_columns
    )

    return engineered_df


# ============================================================
# START
# ============================================================

print("\n==========================================")
print(" FEATURE ENGINEERING EXPERIMENT")
print("==========================================")


# ============================================================
# CHECK DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset not found.")
    print("Expected:")
    print(DATASET_PATH)

    raise SystemExit


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Original rows: {len(df)}"
)

print(
    f"Original columns: {len(df.columns)}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

df = df.reset_index(
    drop=True
)

after = len(df)

print(
    f"Duplicate rows removed: {before - after}"
)

print(
    f"Rows after duplicate removal: {after}"
)


# ============================================================
# CHECK LABEL
# ============================================================

if "label" not in df.columns:

    print(
        "\nERROR: 'label' column not found."
    )

    raise SystemExit


# ============================================================
# DISPLAY CLASSES
# ============================================================

print("\nGesture classes:")

for gesture in sorted(
    df["label"].unique()
):

    count = (
        df["label"] == gesture
    ).sum()

    print(
        f" - {gesture}: {count}"
    )


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

X_raw = df.drop(
    "label",
    axis=1
)

y = df["label"]


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

print(
    "\nCreating engineered features..."
)

X_engineered = create_engineered_features(
    X_raw
)

print(
    "✓ Feature engineering completed."
)

print(
    f"Original feature count: {X_raw.shape[1]}"
)

print(
    f"Engineered feature count: "
    f"{X_engineered.shape[1]}"
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_values = (
    X_engineered.isnull()
    .sum()
    .sum()
)

if missing_values > 0:

    print(
        "\nERROR: Missing values found:"
    )

    print(
        missing_values
    )

    raise SystemExit

else:

    print(
        "✓ No missing values."
    )


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

infinite_values = np.isinf(
    X_engineered.values
).sum()

if infinite_values > 0:

    print(
        "\nERROR: Infinite values found:"
    )

    print(
        infinite_values
    )

    raise SystemExit

else:

    print(
        "✓ No infinite values."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_engineered,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\n==========================================")
print(" TRAINING ENGINEERED-FEATURE SVM")
print("==========================================")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# FEATURE SCALING
# ============================================================

print(
    "\nScaling features..."
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "✓ Feature scaling completed."
)


# ============================================================
# TRAIN SVM
# ============================================================

print(
    "\nTraining SVM..."
)

model = SVC(

    kernel="rbf",

    C=10,

    gamma="scale",

    probability=True,

    random_state=RANDOM_STATE
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "✓ Training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nGenerating predictions..."
)

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print(" ENGINEERED MODEL RESULTS")
print("==========================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE ENGINEERED DATASET
# ============================================================

output_dataset = (
    "dataset/gesture_dataset_engineered.csv"
)

engineered_with_label = (
    X_engineered.copy()
)

engineered_with_label["label"] = (
    y.values
)

engineered_with_label.to_csv(

    output_dataset,

    index=False
)

print(
    "\n✓ Engineered dataset saved:"
)

print(
    f"  {output_dataset}"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n==========================================")
print(" EXPERIMENT SUMMARY")
print("==========================================")

print(
    f"Samples used       : {len(df)}"
)

print(
    f"Original features  : {X_raw.shape[1]}"
)

print(
    f"New features       : {X_engineered.shape[1]}"
)

print(
    "Model              : SVM (RBF)"
)

print(
    f"Accuracy           : "
    f"{accuracy * 100:.2f}%"
)

print(
    "\nPrevious raw-coordinate accuracy: 90.83%"
)


# ============================================================
# DECISION
# ============================================================

if accuracy * 100 > 90.83:

    print(
        "\n✓ ENGINEERED FEATURES "
        "IMPROVED ACCURACY."
    )

elif accuracy * 100 < 90.83:

    print(
        "\n⚠ ENGINEERED FEATURES "
        "DID NOT IMPROVE ACCURACY."
    )

else:

    print(
        "\n= Accuracy is approximately "
        "the same."
    )


# ============================================================
# END
# ============================================================

print("\n==========================================")
print(" EXPERIMENT COMPLETED")
print("==========================================")