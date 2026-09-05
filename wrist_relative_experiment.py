import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_PATH = "dataset/gesture_dataset.csv"
LIVE_PATH = "dataset/live_diagnostic.csv"

MODEL_PATH = "trained_model/gesture_svm_wrist_relative.pkl"
SCALER_PATH = "trained_model/scaler_wrist_relative.pkl"

TRAINING_FEATURE_OUTPUT = (
    "dataset/gesture_dataset_wrist_relative.csv"
)

LIVE_FEATURE_OUTPUT = (
    "dataset/live_diagnostic_wrist_relative.csv"
)

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" WRIST-RELATIVE FEATURE EXPERIMENT")
print("==========================================")

print("\nPurpose:")
print("Remove absolute hand position from the")
print("landmark coordinates while preserving")
print("relative hand geometry and natural scale.")

print("\nIMPORTANT:")
print("This is an experimental model.")
print("Your existing 92.36% model will NOT be changed.")


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TRAINING_PATH):

    print(
        f"\nERROR: Training dataset not found:\n"
        f"{TRAINING_PATH}"
    )

    raise SystemExit


if not os.path.exists(LIVE_PATH):

    print(
        f"\nERROR: Live diagnostic dataset not found:\n"
        f"{LIVE_PATH}"
    )

    raise SystemExit


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    TRAINING_PATH
)

live_df = pd.read_csv(
    LIVE_PATH
)

print("✓ Training dataset loaded.")
print("✓ Live diagnostic dataset loaded.")


# ============================================================
# FEATURE NAMES
# ============================================================

raw_feature_names = []

for i in range(21):

    raw_feature_names.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


# ============================================================
# CHECK FEATURES
# ============================================================

for dataset_name, df in [
    ("training", train_df),
    ("live", live_df)
]:

    missing = [
        feature
        for feature in raw_feature_names
        if feature not in df.columns
    ]

    if missing:

        print(
            f"\nERROR: Missing features in "
            f"{dataset_name} dataset:"
        )

        print(missing)

        raise SystemExit


# ============================================================
# CHECK LABEL
# ============================================================

if "label" not in train_df.columns:

    print(
        "\nERROR: Training dataset has no label column."
    )

    raise SystemExit


if "label" not in live_df.columns:

    print(
        "\nERROR: Live dataset has no label column."
    )

    raise SystemExit


# ============================================================
# REMOVE EXACT DUPLICATES FROM TRAINING DATA
# ============================================================

print("\n==========================================")
print(" TRAINING DATA PREPARATION")
print("==========================================")


before_duplicates = len(train_df)


train_df = train_df.drop_duplicates(
    ignore_index=True
)


removed_duplicates = (
    before_duplicates
    - len(train_df)
)


print(
    f"\nOriginal training rows : "
    f"{before_duplicates}"
)

print(
    f"Duplicate rows removed : "
    f"{removed_duplicates}"
)

print(
    f"Unique training rows   : "
    f"{len(train_df)}"
)


# ============================================================
# FEATURE CONVERSION FUNCTION
# ============================================================

def create_wrist_relative_features(df):

    """
    Convert raw MediaPipe landmark coordinates
    into wrist-relative coordinates.

    Landmark 0 = wrist.

    For each landmark:

        relative_x = x_landmark - x_wrist
        relative_y = y_landmark - y_wrist
        relative_z = z_landmark - z_wrist

    No scale normalization is performed.

    This removes absolute hand position while
    preserving natural hand size and geometry.
    """

    raw = (
        df[raw_feature_names]
        .astype(float)
        .to_numpy()
    )


    # --------------------------------------------------------
    # Reshape:
    #
    # samples × 21 landmarks × 3 coordinates
    # --------------------------------------------------------

    landmarks = raw.reshape(
        -1,
        21,
        3
    )


    # --------------------------------------------------------
    # Wrist = landmark 0
    # --------------------------------------------------------

    wrist = landmarks[
        :,
        0,
        :
    ]


    # --------------------------------------------------------
    # Subtract wrist from every landmark
    # --------------------------------------------------------

    relative = (
        landmarks
        - wrist[:, np.newaxis, :]
    )


    # --------------------------------------------------------
    # Flatten back to 63 features
    # --------------------------------------------------------

    relative = relative.reshape(
        -1,
        63
    )


    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    relative_names = []

    for i in range(21):

        relative_names.extend([
            f"rx_{i}",
            f"ry_{i}",
            f"rz_{i}"
        ])


    relative_df = pd.DataFrame(
        relative,
        columns=relative_names
    )


    return relative_df


# ============================================================
# CREATE FEATURES
# ============================================================

print("\n==========================================")
print(" FEATURE TRANSFORMATION")
print("==========================================")


print(
    "\nCreating wrist-relative training features..."
)


X_train = create_wrist_relative_features(
    train_df
)


print(
    "✓ Training features created."
)


print(
    "\nCreating wrist-relative live features..."
)


X_live = create_wrist_relative_features(
    live_df
)


print(
    "✓ Live features created."
)


y_train = train_df[
    "label"
].to_numpy()


y_live = live_df[
    "label"
].to_numpy()


# ============================================================
# FEATURE QUALITY CHECK
# ============================================================

print("\n==========================================")
print(" FEATURE QUALITY CHECK")
print("==========================================")


train_missing = (
    X_train
    .isnull()
    .sum()
    .sum()
)


live_missing = (
    X_live
    .isnull()
    .sum()
    .sum()
)


train_infinite = np.isinf(
    X_train.to_numpy()
).sum()


live_infinite = np.isinf(
    X_live.to_numpy()
).sum()


print(
    f"\nTraining missing values : "
    f"{train_missing}"
)

print(
    f"Live missing values     : "
    f"{live_missing}"
)

print(
    f"Training infinite       : "
    f"{train_infinite}"
)

print(
    f"Live infinite           : "
    f"{live_infinite}"
)


if (
    train_missing > 0
    or live_missing > 0
    or train_infinite > 0
    or live_infinite > 0
):

    print(
        "\nERROR: Invalid feature values detected."
    )

    raise SystemExit


# ============================================================
# CONVERT TO NUMPY
# ============================================================

# This prevents pandas/Arrow indexing problems
# during train_test_split and sklearn operations.

X_train_array = X_train.to_numpy(
    dtype=float
)

X_live_array = X_live.to_numpy(
    dtype=float
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n==========================================")
print(" TRANSFORMED FEATURE STATISTICS")
print("==========================================")


print(
    f"\nTraining minimum: "
    f"{X_train_array.min():.6f}"
)

print(
    f"Training maximum: "
    f"{X_train_array.max():.6f}"
)

print(
    f"Training mean: "
    f"{X_train_array.mean():.6f}"
)


print(
    f"\nLive minimum: "
    f"{X_live_array.min():.6f}"
)

print(
    f"Live maximum: "
    f"{X_live_array.max():.6f}"
)

print(
    f"Live mean: "
    f"{X_live_array.mean():.6f}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n==========================================")
print(" TRAIN / TEST SPLIT")
print("==========================================")


X_train_split, X_test_split, y_train_split, y_test_split = (

    train_test_split(

        X_train_array,

        y_train,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y_train

    )

)


print(
    f"\nTraining samples: "
    f"{len(X_train_split)}"
)

print(
    f"Internal test samples: "
    f"{len(X_test_split)}"
)


# ============================================================
# SCALING
# ============================================================

print("\n==========================================")
print(" FEATURE SCALING")
print("==========================================")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_split
)


X_test_scaled = scaler.transform(
    X_test_split
)


X_live_scaled = scaler.transform(
    X_live_array
)


print(
    "✓ StandardScaler fitted."
)

print(
    "✓ Training data scaled."
)

print(
    "✓ Internal test data scaled."
)

print(
    "✓ Live diagnostic data scaled."
)


# ============================================================
# TRAIN SVM
# ============================================================

print("\n==========================================")
print(" TRAINING WRIST-RELATIVE SVM")
print("==========================================")


print(
    "\nKernel: RBF"
)

print(
    "C: 10"
)

print(
    "Gamma: scale"
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
    y_train_split
)


print(
    "\n✓ Experimental model trained."
)


# ============================================================
# INTERNAL TEST
# ============================================================

print("\n==========================================")
print(" INTERNAL TEST PERFORMANCE")
print("==========================================")


internal_pred = model.predict(
    X_test_scaled
)


internal_accuracy = accuracy_score(
    y_test_split,
    internal_pred
)


print(
    f"\nInternal test accuracy: "
    f"{internal_accuracy * 100:.2f}%"
)


print(
    "\nClassification report:"
)


print(
    classification_report(
        y_test_split,
        internal_pred,
        digits=4
    )
)


# ============================================================
# LIVE HOLDOUT EVALUATION
# ============================================================

print("\n==========================================")
print(" LIVE HOLDOUT EVALUATION")
print("==========================================")


print(
    "\nEvaluating on the 600 live samples..."
)


live_pred = model.predict(
    X_live_scaled
)


live_accuracy = accuracy_score(
    y_live,
    live_pred
)


print(
    f"\nLIVE ACCURACY: "
    f"{live_accuracy * 100:.2f}%"
)


# ============================================================
# LIVE CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print(" LIVE CLASSIFICATION REPORT")
print("==========================================")


live_report = classification_report(

    y_live,

    live_pred,

    digits=4

)


print(
    "\n" + live_report
)


# ============================================================
# LIVE CONFUSION MATRIX
# ============================================================

print("==========================================")
print(" LIVE CONFUSION MATRIX")
print("==========================================")


labels = sorted(
    np.unique(y_live)
)


live_cm = confusion_matrix(

    y_live,

    live_pred,

    labels=labels

)


print(
    "\nLabels:"
)

print(
    labels
)


print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted\n"
)

print(
    live_cm
)


# ============================================================
# PER-GESTURE LIVE ACCURACY
# ============================================================

print("\n==========================================")
print(" PER-GESTURE LIVE ACCURACY")
print("==========================================")


for gesture in labels:

    mask = (
        y_live == gesture
    )


    total = mask.sum()


    correct = (
        live_pred[mask] == gesture
    ).sum()


    gesture_accuracy = (
        correct
        / total
        * 100
    )


    print(
        f"{gesture:12s} : "
        f"{gesture_accuracy:6.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# LIVE ERROR ANALYSIS
# ============================================================

print("\n==========================================")
print(" LIVE ERROR ANALYSIS")
print("==========================================")


wrong_mask = (
    y_live != live_pred
)


wrong_count = (
    wrong_mask.sum()
)


correct_count = (
    len(y_live)
    - wrong_count
)


print(
    f"\nCorrect predictions: "
    f"{correct_count}"
)


print(
    f"Incorrect predictions: "
    f"{wrong_count}"
)


if wrong_count > 0:

    errors = pd.DataFrame({

        "Actual":
            y_live[wrong_mask],

        "Predicted":
            live_pred[wrong_mask]

    })


    error_pairs = (

        errors
        .value_counts()
        .reset_index(
            name="Count"
        )

    )


    print(
        "\nMost common errors:"
    )


    for _, row in error_pairs.head(15).iterrows():

        print(
            f"  {str(row['Actual']):12s}"
            f" → "
            f"{str(row['Predicted']):12s}"
            f": {row['Count']}"
        )


# ============================================================
# COMPARE AGAINST CURRENT MODELS
# ============================================================

print("\n==========================================")
print(" MODEL COMPARISON")
print("==========================================")


print(
    "\nPrevious results:"
)


print(
    "Original raw SVM      : 90.83%"
)

print(
    "Augmented raw SVM     : 92.36%"
)

print(
    "Previous normalized   : 81.66%"
)

print(
    "Current live baseline :  9.67%"
)


print(
    "\nWrist-relative model:"
)


print(
    f"Internal test        : "
    f"{internal_accuracy * 100:.2f}%"
)


print(
    f"Live holdout         : "
    f"{live_accuracy * 100:.2f}%"
)


# ============================================================
# IMPROVEMENT CALCULATION
# ============================================================

live_baseline = 9.67


improvement = (
    live_accuracy * 100
    - live_baseline
)


print(
    f"\nImprovement over 9.67% "
    f"live baseline: "
    f"{improvement:+.2f} percentage points"
)


# ============================================================
# DECISION
# ============================================================

print("\n==========================================")
print(" EXPERIMENT DECISION")
print("==========================================")


if live_accuracy * 100 >= 80:

    print(
        "\n✓ VERY PROMISING"
    )

    print(
        "Wrist-relative features substantially"
    )

    print(
        "improve live generalization."
    )

    print(
        "\nThis feature representation should"
    )

    print(
        "be considered for the next development"
    )

    print(
        "stage."
    )


elif live_accuracy * 100 >= 50:

    print(
        "\n⚠ PARTIAL IMPROVEMENT"
    )

    print(
        "Wrist-relative features help, but"
    )

    print(
        "live robustness is still insufficient."
    )

    print(
        "\nFurther feature/domain experiments"
    )

    print(
        "are recommended."
    )


else:

    print(
        "\n⚠ INSUFFICIENT IMPROVEMENT"
    )

    print(
        "Wrist-relative coordinates alone do"
    )

    print(
        "not solve the training/live mismatch."
    )

    print(
        "\nDo not replace the current model."
    )

    print(
        "We should investigate the collection"
    )

    print(
        "pipeline and landmark representation"
    )

    print(
        "before retraining."
    )


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

print("\n==========================================")
print(" SAVING EXPERIMENTAL MODEL")
print("==========================================")


joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    scaler,
    SCALER_PATH
)


print(
    f"\nModel saved to:"
)

print(
    MODEL_PATH
)


print(
    "\nScaler saved to:"
)

print(
    SCALER_PATH
)


# ============================================================
# SAVE TRANSFORMED DATA
# ============================================================

train_output = X_train.copy()

train_output[
    "label"
] = y_train


live_output = X_live.copy()

live_output[
    "label"
] = y_live


train_output.to_csv(
    TRAINING_FEATURE_OUTPUT,
    index=False
)


live_output.to_csv(
    LIVE_FEATURE_OUTPUT,
    index=False
)


print(
    "\nTransformed training data saved to:"
)

print(
    TRAINING_FEATURE_OUTPUT
)


print(
    "\nTransformed live data saved to:"
)

print(
    LIVE_FEATURE_OUTPUT
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print(" EXPERIMENT COMPLETE")
print("==========================================")


print(
    f"\nFinal internal accuracy: "
    f"{internal_accuracy * 100:.2f}%"
)


print(
    f"Final live accuracy: "
    f"{live_accuracy * 100:.2f}%"
)


print(
    "\nYour original 92.36% model remains"
)

print(
    "completely untouched."
)


print(
    "\n==========================================\n"
)