# ============================================================
# AI HAND GESTURE RECOGNITION - V2 MODEL TRAINING
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


# ============================================================
# PATHS
# ============================================================

# train_model_v2.py is inside:
# AI_Hand_Gesture_Presentation/dataset/
#
# parent       -> dataset
# parent.parent -> AI_Hand_Gesture_Presentation

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "gesture_dataset_v2.csv"
MODEL_DIR = BASE_DIR / "trained_model"
RESULTS_PATH = BASE_DIR / "dataset" / "v2_evaluation_results.txt"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 75)
print("AI HAND GESTURE RECOGNITION - V2 MODEL TRAINING")
print("=" * 75)


# ============================================================
# 1. DATASET CHECK
# ============================================================

print("\n[1] DATASET CHECK")

if not DATASET_PATH.exists():
    print("\nERROR: Dataset not found!")
    print("\nExpected:")
    print(DATASET_PATH)
    raise SystemExit

print("\nDataset found:")
print(DATASET_PATH)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[2] LOADING DATASET")

try:
    df = pd.read_csv(DATASET_PATH)
except Exception as e:
    print("\nERROR while reading dataset:")
    print(e)
    raise SystemExit

print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 3. COLUMN CHECK
# ============================================================

print("\n[3] COLUMN CHECK")

if "label" not in df.columns:
    print("ERROR: 'label' column not found.")
    raise SystemExit

if "session_id" not in df.columns:
    print("ERROR: 'session_id' column not found.")
    raise SystemExit


# Find the 63 landmark features
feature_columns = [
    col for col in df.columns
    if col.startswith("x_")
    or col.startswith("y_")
    or col.startswith("z_")
]

print(f"Feature columns found: {len(feature_columns)}")

if len(feature_columns) != 63:
    print("\nERROR: Expected exactly 63 feature columns.")
    print(f"Found: {len(feature_columns)}")
    raise SystemExit

print("63 landmark features confirmed.")


# ============================================================
# 4. BASIC DATA VALIDATION
# ============================================================

print("\n[4] DATA VALIDATION")

missing_values = df[feature_columns + ["label", "session_id"]].isnull().sum().sum()

print(f"Missing values: {missing_values}")

if missing_values > 0:
    print("ERROR: Missing values found.")
    raise SystemExit


X = df[feature_columns].copy()
y = df["label"].copy()

# Convert feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().sum().sum() > 0:
    print("ERROR: Non-numeric or invalid feature values found.")
    raise SystemExit


# Check finite values
if not np.isfinite(X.to_numpy()).all():
    print("ERROR: Non-finite values found.")
    raise SystemExit

print("All feature values are valid.")


# ============================================================
# 5. GESTURE DISTRIBUTION
# ============================================================

print("\n[5] GESTURE DISTRIBUTION")

gesture_counts = y.value_counts().sort_index()

for gesture, count in gesture_counts.items():
    print(f"{gesture:<15} {count}")

print(f"\nTotal samples: {len(df)}")
print(f"Number of gestures: {y.nunique()}")


# ============================================================
# 6. SESSION INFORMATION
# ============================================================

print("\n[6] SESSION INFORMATION")

session_table = (
    df.groupby(["session_id", "label"])
    .size()
    .reset_index(name="samples")
)

print(session_table.to_string(index=False))

print(f"\nTotal sessions: {df['session_id'].nunique()}")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n[7] TRAIN / TEST SPLIT")

print("Using stratified 80/20 split...")
print("Random state: 42")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# IMPORTANT NOTE
# ============================================================

print("\nNOTE:")
print("Most gestures currently have only one recording session.")
print("Therefore, this is an internal stratified benchmark,")
print("NOT a true unseen-session generalization test.")

print("\nThe model will later be tested on a NEW live session.")


# ============================================================
# 8. FEATURE SCALING
# ============================================================

print("\n[8] FEATURE SCALING")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler fitted successfully.")


# ============================================================
# 9. TRAIN SVM
# ============================================================

print("\n[9] TRAINING SVM")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("SVM training completed.")


# ============================================================
# 10. PREDICTION
# ============================================================

print("\n[10] MAKING PREDICTIONS")

y_pred = model.predict(X_test_scaled)

print("Predictions completed.")


# ============================================================
# 11. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 75)
print("V2 MODEL ACCURACY")
print("=" * 75)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    digits=4
)

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{label}" for label in labels],
    columns=[f"Pred_{label}" for label in labels]
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm_df)


# ============================================================
# 14. SAVE MODEL
# ============================================================

print("\n[14] SAVING MODEL")

model_path = MODEL_DIR / "gesture_svm_v2.pkl"
scaler_path = MODEL_DIR / "scaler_v2.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModel saved:")
print(model_path)

print(f"\nScaler saved:")
print(scaler_path)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

print("\n[15] SAVING EVALUATION RESULTS")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:

    f.write("=" * 75 + "\n")
    f.write("AI HAND GESTURE RECOGNITION - V2 MODEL EVALUATION\n")
    f.write("=" * 75 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 75 + "\n")
    f.write(f"Dataset: {DATASET_PATH}\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Total features: {len(feature_columns)}\n")
    f.write(f"Gestures: {y.nunique()}\n")
    f.write(f"Sessions: {df['session_id'].nunique()}\n\n")

    f.write("GESTURE DISTRIBUTION\n")
    f.write("-" * 75 + "\n")
    f.write(gesture_counts.to_string())
    f.write("\n\n")

    f.write("TRAIN / TEST SPLIT\n")
    f.write("-" * 75 + "\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")
    f.write("Split: Stratified 80/20\n")
    f.write("Random state: 42\n\n")

    f.write("MODEL\n")
    f.write("-" * 75 + "\n")
    f.write("Algorithm: Support Vector Machine\n")
    f.write("Kernel: RBF\n")
    f.write("C: 10\n")
    f.write("Gamma: scale\n")
    f.write("Scaler: StandardScaler\n\n")

    f.write("ACCURACY\n")
    f.write("-" * 75 + "\n")
    f.write(f"{accuracy * 100:.2f}%\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 75 + "\n")
    f.write(report)
    f.write("\n")

    f.write("CONFUSION MATRIX\n")
    f.write("-" * 75 + "\n")
    f.write(cm_df.to_string())
    f.write("\n\n")

    f.write("IMPORTANT NOTE\n")
    f.write("-" * 75 + "\n")
    f.write(
        "This is an internal stratified random-split benchmark. "
        "Because most gestures currently have only one recording session, "
        "the result should not be interpreted as true unseen-session "
        "generalization. A separate live-session evaluation is required.\n"
    )

print(f"\nResults saved:")
print(RESULTS_PATH)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("V2 TRAINING COMPLETE")
print("=" * 75)

print(f"\nV2 Accuracy: {accuracy * 100:.2f}%")

print("\nFiles created:")
print(f"  - {model_path}")
print(f"  - {scaler_path}")
print(f"  - {RESULTS_PATH}")

print("\nNEXT STEP:")
print("Test the V2 model on a completely NEW live webcam session.")

print("\n" + "=" * 75)