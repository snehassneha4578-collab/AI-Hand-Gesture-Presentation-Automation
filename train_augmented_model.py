import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/gesture_dataset.csv"

MODEL_DIR = "trained_model"

AUGMENTATION_FACTOR = 3

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" AUGMENTED RAW-COORDINATE SVM")
print("==========================================")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset not found.")
    print(DATASET_PATH)

    raise SystemExit


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Original rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

df = df.reset_index(drop=True)

after = len(df)

print(
    f"Duplicate rows removed: {before - after}"
)

print(
    f"Rows after duplicate removal: {after}"
)


# ============================================================
# CHECK FEATURES
# ============================================================

expected_features = []

for i in range(21):

    expected_features.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


missing_features = [
    feature
    for feature in expected_features
    if feature not in df.columns
]


if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(
            f" - {feature}"
        )

    raise SystemExit


# ============================================================
# SEPARATE X AND Y
# ============================================================

X = df[expected_features].copy()

y = df["label"].copy()


# ============================================================
# DISPLAY CLASSES
# ============================================================

print("\nGesture classes:")

for gesture in sorted(
    y.unique()
):

    count = (
        y == gesture
    ).sum()

    print(
        f" - {gesture}: {count}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n==========================================")
print(" TRAIN / TEST SPLIT")
print("==========================================")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    f"Original training samples: {len(X_train)}"
)

print(
    f"Original testing samples : {len(X_test)}"
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

print("\n==========================================")
print(" TRAINING-DATA AUGMENTATION")
print("==========================================")


rng = np.random.default_rng(
    RANDOM_STATE
)


# Convert to NumPy

X_train_array = X_train.values.astype(
    np.float64
)

y_train_array = y_train.values


augmented_X = [
    X_train_array
]

augmented_y = [
    y_train_array
]


# ------------------------------------------------------------
# Augmentation strategy
# ------------------------------------------------------------
#
# We apply small variations to the hand landmarks:
#
# 1. Small global translation
# 2. Small scale change
# 3. Very small coordinate noise
#
# These simulate natural changes in webcam usage.
#
# IMPORTANT:
# Only the TRAINING set is augmented.
#
# The TEST set remains completely untouched.
# ------------------------------------------------------------


for augmentation_number in range(
    AUGMENTATION_FACTOR
):

    print(
        f"Creating augmentation "
        f"{augmentation_number + 1}/"
        f"{AUGMENTATION_FACTOR}..."
    )


    augmented_samples = []


    for sample in X_train_array:

        # Reshape:
        #
        # 63 values
        # ↓
        # 21 landmarks × 3 coordinates

        landmarks = sample.reshape(
            21,
            3
        ).copy()


        # ----------------------------------------------------
        # Random translation
        # ----------------------------------------------------

        translation_x = rng.normal(
            0,
            0.008
        )

        translation_y = rng.normal(
            0,
            0.008
        )

        translation_z = rng.normal(
            0,
            0.003
        )


        landmarks[:, 0] += (
            translation_x
        )

        landmarks[:, 1] += (
            translation_y
        )

        landmarks[:, 2] += (
            translation_z
        )


        # ----------------------------------------------------
        # Random scale
        # ----------------------------------------------------

        scale = rng.uniform(
            0.95,
            1.05
        )


        # Use wrist as approximate center

        wrist = landmarks[0].copy()


        landmarks = (
            wrist
            + (
                landmarks - wrist
            ) * scale
        )


        # ----------------------------------------------------
        # Very small coordinate noise
        # ----------------------------------------------------

        noise = rng.normal(
            0,
            0.002,
            size=landmarks.shape
        )


        landmarks += noise


        # ----------------------------------------------------
        # Keep reasonable coordinate range
        # ----------------------------------------------------

        landmarks[:, 0] = np.clip(
            landmarks[:, 0],
            0.0,
            1.0
        )

        landmarks[:, 1] = np.clip(
            landmarks[:, 1],
            0.0,
            1.0
        )


        augmented_samples.append(
            landmarks.reshape(63)
        )


    augmented_X.append(
        np.array(augmented_samples)
    )

    augmented_y.append(
        y_train_array.copy()
    )


# ============================================================
# COMBINE ORIGINAL + AUGMENTED
# ============================================================

X_train_augmented = np.vstack(
    augmented_X
)

y_train_augmented = np.concatenate(
    augmented_y
)


print("\nAugmentation completed.")

print(
    f"Original training samples : "
    f"{len(X_train_array)}"
)

print(
    f"Augmented training samples: "
    f"{len(X_train_augmented)}"
)


# ============================================================
# STANDARD SCALER
# ============================================================

print("\n==========================================")
print(" FEATURE SCALING")
print("==========================================")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_augmented
)

X_test_scaled = scaler.transform(
    X_test
)


print(
    "✓ Scaling completed."
)


# ============================================================
# TRAIN SVM
# ============================================================

print("\n==========================================")
print(" TRAINING AUGMENTED SVM")
print("==========================================")


model = SVC(

    kernel="rbf",

    C=10,

    gamma="scale",

    probability=True,

    random_state=RANDOM_STATE
)


print(
    "Training model..."
)


model.fit(

    X_train_scaled,

    y_train_augmented

)


print(
    "✓ Training completed."
)


# ============================================================
# TEST
# ============================================================

print("\nGenerating predictions...")


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
print(" AUGMENTED MODEL RESULTS")
print("==========================================")


print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
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
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


AUGMENTED_MODEL_PATH = (
    MODEL_DIR
    + "/gesture_svm_augmented.pkl"
)


AUGMENTED_SCALER_PATH = (
    MODEL_DIR
    + "/scaler_augmented.pkl"
)


joblib.dump(
    model,
    AUGMENTED_MODEL_PATH
)


joblib.dump(
    scaler,
    AUGMENTED_SCALER_PATH
)


print("\n==========================================")
print(" FILES SAVED")
print("==========================================")


print(
    f"\n✓ Model:"
)

print(
    f"  {AUGMENTED_MODEL_PATH}"
)


print(
    f"\n✓ Scaler:"
)

print(
    f"  {AUGMENTED_SCALER_PATH}"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n==========================================")
print(" FINAL COMPARISON")
print("==========================================")


print(
    "\nPrevious raw-coordinate SVM:"
)

print(
    "  Accuracy = 90.83%"
)


print(
    "\nNormalized-coordinate SVM:"
)

print(
    "  Accuracy = 81.66%"
)


print(
    "\nEngineered-feature SVM:"
)

print(
    "  Accuracy = 88.21%"
)


print(
    "\nAugmented raw-coordinate SVM:"
)

print(
    f"  Accuracy = "
    f"{accuracy * 100:.2f}%"
)


if accuracy * 100 > 90.83:

    improvement = (
        accuracy * 100
        - 90.83
    )

    print(
        f"\n✓ IMPROVED by "
        f"{improvement:.2f} percentage points."
    )


elif accuracy * 100 < 90.83:

    decrease = (
        90.83
        - accuracy * 100
    )

    print(
        f"\n⚠ LOWER by "
        f"{decrease:.2f} percentage points."
    )


else:

    print(
        "\n= Same accuracy as baseline."
    )


# ============================================================
# END
# ============================================================

print("\n==========================================")
print(" AUGMENTATION EXPERIMENT COMPLETED")
print("==========================================")