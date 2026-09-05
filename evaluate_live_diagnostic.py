import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/live_diagnostic.csv"

MODEL_PATH = "trained_model/gesture_svm_augmented.pkl"

SCALER_PATH = "trained_model/scaler_augmented.pkl"


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" LIVE DIAGNOSTIC MODEL EVALUATION")
print("==========================================")

print("\nThis test evaluates the existing")
print("augmented SVM on your NEW live samples.")

print("\nIMPORTANT:")
print("The model will NOT be retrained.")
print("The model will NOT be modified.")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH
]


for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            f"\nERROR: File not found:"
            f"\n{file_path}"
        )

        raise SystemExit


# ============================================================
# LOAD LIVE DATASET
# ============================================================

print("\nLoading live diagnostic dataset...")

df = pd.read_csv(
    DATASET_PATH
)


print("✓ Dataset loaded.")

print(
    f"Rows: {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = []

for i in range(21):

    feature_names.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [

    feature
    for feature in feature_names
    if feature not in df.columns

]


if missing_features:

    print(
        "\nERROR: Missing features:"
    )

    print(
        missing_features
    )

    raise SystemExit


# ============================================================
# CHECK LABEL
# ============================================================

if "label" not in df.columns:

    print(
        "\nERROR: 'label' column not found."
    )

    raise SystemExit


# ============================================================
# DISPLAY DATASET
# ============================================================

print("\n==========================================")
print(" DATASET INFORMATION")
print("==========================================")

print(
    f"\nTotal samples: {len(df)}"
)

print(
    f"Total features: {len(feature_names)}"
)

print(
    f"Labels: {df['label'].nunique()}"
)


print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_values = (
    df[feature_names]
    .isnull()
    .sum()
    .sum()
)


print(
    f"\nMissing feature values: {missing_values}"
)


if missing_values > 0:

    print(
        "\nERROR: Missing values detected."
    )

    raise SystemExit


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

X_check = df[
    feature_names
].values


infinite_values = np.isinf(
    X_check
).sum()


print(
    f"Infinite feature values: {infinite_values}"
)


if infinite_values > 0:

    print(
        "\nERROR: Infinite values detected."
    )

    raise SystemExit


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[
    feature_names
]


y = df[
    "label"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print(" LOADING MODEL")
print("==========================================")


print(
    "\nLoading augmented SVM..."
)

model = joblib.load(
    MODEL_PATH
)


print(
    "✓ Augmented SVM loaded."
)


# ============================================================
# LOAD SCALER
# ============================================================

print(
    "\nLoading scaler..."
)

scaler = joblib.load(
    SCALER_PATH
)


print(
    "✓ Scaler loaded."
)


# ============================================================
# SCALE LIVE FEATURES
# ============================================================

print(
    "\nScaling live features..."
)


X_scaled = scaler.transform(
    X.values
)


print(
    "✓ Scaling complete."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nRunning predictions..."
)


y_pred = model.predict(
    X_scaled
)


print(
    "✓ Predictions complete."
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y,
    y_pred
)


accuracy_percent = (
    accuracy * 100
)


print("\n==========================================")
print(" OVERALL LIVE ACCURACY")
print("==========================================")


print(
    f"\nAccuracy: "
    f"{accuracy_percent:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print(" CLASSIFICATION REPORT")
print("==========================================")


report = classification_report(

    y,

    y_pred,

    digits=4

)


print(
    "\n" + report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("==========================================")
print(" CONFUSION MATRIX")
print("==========================================")


labels = sorted(
    y.unique()
)


cm = confusion_matrix(

    y,

    y_pred,

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
    cm
)


# ============================================================
# PER-GESTURE ACCURACY
# ============================================================

print("\n==========================================")
print(" PER-GESTURE ACCURACY")
print("==========================================")


for gesture in labels:

    actual_mask = (
        y == gesture
    )


    total = actual_mask.sum()


    correct = (
        (y_pred[actual_mask] == gesture)
        .sum()
    )


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
# PREDICTION DISTRIBUTION
# ============================================================

print("\n==========================================")
print(" PREDICTION DISTRIBUTION")
print("==========================================")


prediction_counts = (
    pd.Series(y_pred)
    .value_counts()
)


for gesture in labels:

    count = prediction_counts.get(
        gesture,
        0
    )


    print(
        f"{gesture:12s} : "
        f"{count}"
    )


# ============================================================
# MISCLASSIFICATION ANALYSIS
# ============================================================

print("\n==========================================")
print(" MISCLASSIFICATION ANALYSIS")
print("==========================================")


incorrect_mask = (
    y != y_pred
)


incorrect_count = (
    incorrect_mask.sum()
)


print(
    f"\nIncorrect predictions: "
    f"{incorrect_count}"
)


print(
    f"Correct predictions: "
    f"{len(y) - incorrect_count}"
)


if incorrect_count > 0:

    error_pairs = pd.DataFrame({

        "Actual": y[incorrect_mask],

        "Predicted": y_pred[incorrect_mask]

    })


    pair_counts = (

        error_pairs
        .value_counts()
        .reset_index(
            name="Count"
        )

    )


    print(
        "\nMost common errors:"
    )


    for _, row in pair_counts.iterrows():

        print(

            f"  Actual: "
            f"{row['Actual']:12s}"
            f" → Predicted: "
            f"{row['Predicted']:12s}"
            f" : {row['Count']}"

        )


# ============================================================
# GESTURE CONFUSION SUMMARY
# ============================================================

print("\n==========================================")
print(" GESTURE CONFUSION SUMMARY")
print("==========================================")


for gesture in labels:

    actual_mask = (
        y == gesture
    )


    wrong_predictions = (
        y_pred[actual_mask]
    )


    wrong_predictions = (
        wrong_predictions[
            wrong_predictions != gesture
        ]
    )


    if len(wrong_predictions) == 0:

        print(
            f"\n{gesture.upper()}: "
            f"No misclassifications ✓"
        )

        continue


    confusion_counts = (
        pd.Series(
            wrong_predictions
        )
        .value_counts()
    )


    print(
        f"\n{gesture.upper()}:"
    )


    for predicted, count in (
        confusion_counts.items()
    ):

        print(
            f"  → {predicted}: "
            f"{count}"
        )


# ============================================================
# COMPARISON WITH BASELINE
# ============================================================

baseline_accuracy = 90.83

augmented_accuracy = 92.36


print("\n==========================================")
print(" COMPARISON WITH PREVIOUS RESULTS")
print("==========================================")


print(
    f"\nOriginal raw SVM: "
    f"{baseline_accuracy:.2f}%"
)


print(
    f"Augmented SVM:    "
    f"{augmented_accuracy:.2f}%"
)


print(
    f"Live diagnostic:  "
    f"{accuracy_percent:.2f}%"
)


live_vs_augmented = (

    accuracy_percent
    - augmented_accuracy

)


print(
    f"\nLive vs augmented model: "
    f"{live_vs_augmented:+.2f} percentage points"
)


# ============================================================
# AUTOMATIC INTERPRETATION
# ============================================================

print("\n==========================================")
print(" AUTOMATIC INTERPRETATION")
print("==========================================")


if accuracy_percent >= 90:

    print(
        "\n✓ EXCELLENT LIVE PERFORMANCE"
    )

    print(
        "The classifier generalizes well to"
    )

    print(
        "new webcam samples."
    )

    print(
        "\nThe main remaining issue is likely"
    )

    print(
        "real-time temporal stabilization,"
    )

    print(
        "not the ML classifier."
    )


elif accuracy_percent >= 80:

    print(
        "\n✓ GOOD LIVE PERFORMANCE"
    )

    print(
        "The classifier is reasonably"
    )

    print(
        "generalizing to live data."
    )

    print(
        "\nSome additional robustness work"
    )

    print(
        "may improve real-time recognition."
    )


elif accuracy_percent >= 70:

    print(
        "\n⚠ MODERATE LIVE PERFORMANCE"
    )

    print(
        "There is a noticeable gap between"
    )

    print(
        "training performance and live data."
    )

    print(
        "\nWe should investigate feature"
    )

    print(
        "representation and gesture variation."
    )


else:

    print(
        "\n⚠ LOW LIVE PERFORMANCE"
    )

    print(
        "The current model does not"
    )

    print(
        "generalize well to these webcam samples."
    )

    print(
        "\nFurther ML/data improvements"
    )

    print(
        "will likely be necessary."
    )


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print(" EVALUATION COMPLETE")
print("==========================================")


print(
    f"\nFinal live accuracy: "
    f"{accuracy_percent:.2f}%"
)


print(
    "\nModel files were not changed."
)


print(
    "==========================================\n"
)