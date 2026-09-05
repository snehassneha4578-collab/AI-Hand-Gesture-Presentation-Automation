import os
import joblib
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
# AI HAND GESTURE ML MODEL TRAINING
# RAW MEDIAPIPE LANDMARK DATASET
# ============================================================

print("\n==========================================")
print(" AI HAND GESTURE ML MODEL TRAINING")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

BASE_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)


# IMPORTANT:
# Use the ORIGINAL raw-coordinate dataset.
DATASET_PATH = os.path.join(
    BASE_FOLDER,
    "dataset",
    "gesture_dataset.csv"
)


MODEL_FOLDER = os.path.join(
    BASE_FOLDER,
    "trained_model"
)


MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "gesture_svm.pkl"
)


SCALER_PATH = os.path.join(
    MODEL_FOLDER,
    "scaler.pkl"
)


# ============================================================
# 2. CREATE MODEL FOLDER
# ============================================================

if not os.path.exists(MODEL_FOLDER):

    os.makedirs(MODEL_FOLDER)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset not found!")

    print("\nExpected location:")

    print(DATASET_PATH)

    exit()


print("Loading ORIGINAL raw-coordinate dataset...")

print(DATASET_PATH)


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)


print("\nDataset loaded successfully.")


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\n------------------------------------------")
print("DATASET INFORMATION")
print("------------------------------------------")


print(
    "Total rows:",
    len(df)
)


print(
    "Total columns:",
    len(df.columns)
)


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)


df = df.drop_duplicates()


after_duplicates = len(df)


duplicates_removed = (
    before_duplicates
    - after_duplicates
)


print(
    "\nDuplicate rows removed:",
    duplicates_removed
)


print(
    "Rows remaining:",
    len(df)
)


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

missing_values = (
    df.isnull().sum().sum()
)


print(
    "Missing values:",
    missing_values
)


if missing_values > 0:

    print(
        "\nERROR: Dataset contains missing values."
    )

    print(
        "Please clean the dataset before training."
    )

    exit()


# ============================================================
# 8. CHECK LABEL
# ============================================================

if "label" not in df.columns:

    print(
        "\nERROR: label column not found!"
    )

    exit()


# ============================================================
# 9. SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(
    "label",
    axis=1
)


y = df["label"]


# ============================================================
# 10. GESTURE DISTRIBUTION
# ============================================================

print("\n------------------------------------------")
print("GESTURE DISTRIBUTION")
print("------------------------------------------")


print(
    y.value_counts()
)


# ============================================================
# 11. CHECK FEATURE COUNT
# ============================================================

print(
    "\nNumber of features:",
    X.shape[1]
)


if X.shape[1] != 63:

    print("\nWARNING:")

    print(
        "Expected 63 MediaPipe landmark features."
    )

    print(
        "Found:",
        X.shape[1]
    )

else:

    print(
        "✓ Correct 63-feature structure."
    )


# ============================================================
# 12. CHECK FEATURE ORDER
# ============================================================

expected_features = []


for i in range(21):

    expected_features.append(
        f"x_{i}"
    )

    expected_features.append(
        f"y_{i}"
    )

    expected_features.append(
        f"z_{i}"
    )


if list(X.columns) == expected_features:

    print(
        "✓ Feature order verified."
    )

else:

    print(
        "\nWARNING: Feature order is different."
    )


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

print("\n------------------------------------------")
print("TRAIN / TEST SPLIT")
print("------------------------------------------")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "Training samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 14. FEATURE SCALING
# ============================================================

print("\n------------------------------------------")
print("FEATURE SCALING")
print("------------------------------------------")


print(
    "Scaling raw MediaPipe coordinates..."
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
# 15. CREATE SVM
# ============================================================

print("\n------------------------------------------")
print("SVM MODEL")
print("------------------------------------------")


model = SVC(

    kernel="rbf",

    C=10,

    gamma="scale",

    probability=True,

    random_state=42

)


print(
    "Kernel      : RBF"
)


print(
    "C           : 10"
)


print(
    "Gamma       : scale"
)


print(
    "Probability : True"
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\n------------------------------------------")
print("MODEL TRAINING")
print("------------------------------------------")


print(
    "Training started..."
)


model.fit(

    X_train_scaled,

    y_train

)


print(
    "✓ Training completed successfully!"
)


# ============================================================
# 17. TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)


y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# 18. ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n==========================================")
print(" MODEL PERFORMANCE")
print("==========================================")


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n------------------------------------------")
print("CLASSIFICATION REPORT")
print("------------------------------------------")


print(

    classification_report(

        y_test,

        y_pred

    )

)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

print("\n------------------------------------------")
print("CONFUSION MATRIX")
print("------------------------------------------")


labels = sorted(
    y.unique()
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=labels

)


print(
    "\nGestures:"
)


print(
    labels
)


print(
    "\nMatrix:"
)


print(
    cm
)


# ============================================================
# 21. SAVE MODEL
# ============================================================

print("\n------------------------------------------")
print("SAVING MODEL")
print("------------------------------------------")


joblib.dump(

    model,

    MODEL_PATH

)


joblib.dump(

    scaler,

    SCALER_PATH

)


print(
    "\nSVM model saved to:"
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
# 22. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print(" TRAINING COMPLETE")
print("==========================================")


print(
    "\nDataset samples used:",
    len(df)
)


print(
    "Features:",
    X.shape[1]
)


print(
    "Gestures:",
    len(labels)
)


print(
    "Training samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print(
    "\nGestures recognized by model:"
)


for gesture in labels:

    print(
        " -",
        gesture
    )


print(
    "\nSaved files:"
)


print(
    " - trained_model/gesture_svm.pkl"
)


print(
    " - trained_model/scaler.pkl"
)


print("\n==========================================")
print(" RAW-COORDINATE MODEL READY")
print("==========================================\n")