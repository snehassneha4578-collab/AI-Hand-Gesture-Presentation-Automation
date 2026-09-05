import pandas as pd
import os


# ============================================================
# DATASET INSPECTION
# ============================================================

print("\n==========================================")
print(" AI HAND GESTURE DATASET INSPECTION")
print("==========================================\n")


# ============================================================
# 1. DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(
    BASE_DIR,
    "dataset",
    "gesture_dataset.csv"
)


print("Dataset path:")
print(dataset_path)


# ============================================================
# 2. CHECK FILE
# ============================================================

if not os.path.exists(dataset_path):

    print("\nERROR: Dataset file not found!")

    exit()


print("\nDataset file found.")


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(dataset_path)


print("\nDataset loaded successfully.")


# ============================================================
# 4. SHAPE
# ============================================================

rows, columns = df.shape

print("\n==========================================")
print(" DATASET SIZE")
print("==========================================")

print("Rows:", rows)

print("Columns:", columns)


# ============================================================
# 5. COLUMN CHECK
# ============================================================

print("\n==========================================")
print(" COLUMN INFORMATION")
print("==========================================")

print("First 10 columns:")

print(df.columns[:10].tolist())

print("\nLast 5 columns:")

print(df.columns[-5:].tolist())


# ============================================================
# 6. LABEL COLUMN
# ============================================================

print("\n==========================================")
print(" GESTURE DISTRIBUTION")
print("==========================================")

if "label" not in df.columns:

    print("ERROR: label column not found!")

    exit()


label_counts = df["label"].value_counts()

print(label_counts)


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n==========================================")
print(" MISSING VALUES")
print("==========================================")

total_missing = df.isnull().sum().sum()

print(
    "Total missing values:",
    total_missing
)


if total_missing == 0:

    print("No missing values found. ✅")

else:

    print("Missing values detected. ⚠️")


# ============================================================
# 8. DUPLICATES
# ============================================================

print("\n==========================================")
print(" DUPLICATES")
print("==========================================")

duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_count
)


# ============================================================
# 9. DATA TYPES
# ============================================================

print("\n==========================================")
print(" DATA TYPES")
print("==========================================")

print(
    df.dtypes.tail()
)


# ============================================================
# 10. FEATURE COUNT
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column != "label"
]

print("\n==========================================")
print(" FEATURE CHECK")
print("==========================================")

print(
    "Number of features:",
    len(feature_columns)
)

print(
    "Expected features:",
    21 * 3
)


if len(feature_columns) == 63:

    print("Feature count is correct. ✅")

else:

    print("Feature count is different. ⚠️")


# ============================================================
# 11. UNIQUE LABELS
# ============================================================

print("\n==========================================")
print(" UNIQUE GESTURES")
print("==========================================")

unique_labels = df["label"].unique()

print(
    "Number of gestures:",
    len(unique_labels)
)

print(
    "Gestures:",
    list(unique_labels)
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print(" DATASET SUMMARY")
print("==========================================")

print("Total samples:", len(df))

print("Total features:", len(feature_columns))

print("Total gestures:", len(unique_labels))

print("Missing values:", total_missing)

print("Duplicate rows:", duplicate_count)


print("\n==========================================")
print(" INSPECTION COMPLETE")
print("==========================================")