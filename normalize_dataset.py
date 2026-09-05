import pandas as pd
import numpy as np
import os


# ============================================================
# NORMALIZE HAND LANDMARK DATASET
# ============================================================

print("\n==========================================")
print(" HAND LANDMARK DATASET NORMALIZATION")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

BASE_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_PATH = os.path.join(
    BASE_FOLDER,
    "dataset",
    "gesture_dataset.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_FOLDER,
    "dataset",
    "gesture_dataset_normalized.csv"
)


# ============================================================
# 2. CHECK INPUT
# ============================================================

if not os.path.isfile(INPUT_PATH):

    print("ERROR: Dataset not found!")

    print(INPUT_PATH)

    exit()


print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print("✓ Dataset loaded")

print(
    f"Rows: {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)


# ============================================================
# 3. FEATURE NAMES
# ============================================================

feature_names = []

for i in range(21):

    feature_names.append(
        f"x_{i}"
    )

    feature_names.append(
        f"y_{i}"
    )

    feature_names.append(
        f"z_{i}"
    )


# ============================================================
# 4. CHECK COLUMNS
# ============================================================

missing_columns = [

    column

    for column in feature_names

    if column not in df.columns
]


if missing_columns:

    print("\nERROR: Missing feature columns:")

    print(missing_columns)

    exit()


if "label" not in df.columns:

    print("\nERROR: label column not found!")

    exit()


print("✓ All 63 landmark features found")


# ============================================================
# 5. NORMALIZATION FUNCTION
# ============================================================

def normalize_landmarks(row):

    landmarks = []

    for i in range(21):

        x = float(
            row[f"x_{i}"]
        )

        y = float(
            row[f"y_{i}"]
        )

        z = float(
            row[f"z_{i}"]
        )

        landmarks.append(
            [x, y, z]
        )


    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )


    # ========================================================
    # WRIST = LANDMARK 0
    # ========================================================

    wrist = landmarks[0].copy()


    # ========================================================
    # MAKE COORDINATES WRIST-RELATIVE
    # ========================================================

    normalized = (
        landmarks - wrist
    )


    # ========================================================
    # FIND HAND SCALE
    # ========================================================

    # Distance between wrist and middle-finger MCP.
    #
    # Landmark 9 = middle finger MCP.
    #
    # This provides a stable estimate of hand size.

    palm_size = np.linalg.norm(
        normalized[9]
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if palm_size < 1e-6:

        palm_size = 1.0


    # ========================================================
    # SCALE NORMALIZATION
    # ========================================================

    normalized = (
        normalized / palm_size
    )


    # ========================================================
    # FLATTEN
    # ========================================================

    normalized_features = []

    for landmark in normalized:

        normalized_features.append(
            landmark[0]
        )

        normalized_features.append(
            landmark[1]
        )

        normalized_features.append(
            landmark[2]
        )


    return normalized_features


# ============================================================
# 6. NORMALIZE DATASET
# ============================================================

print("\nNormalizing landmarks...")

normalized_rows = []


for index, row in df.iterrows():

    normalized_features = (
        normalize_landmarks(row)
    )

    normalized_rows.append(
        normalized_features
    )


    if (index + 1) % 500 == 0:

        print(
            f"Processed: "
            f"{index + 1}/{len(df)}"
        )


# ============================================================
# 7. CREATE NORMALIZED DATAFRAME
# ============================================================

normalized_df = pd.DataFrame(

    normalized_rows,

    columns=feature_names
)


# ============================================================
# 8. ADD LABEL
# ============================================================

normalized_df["label"] = (
    df["label"].values
)


# ============================================================
# 9. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = (
    len(normalized_df)
)


normalized_df = (
    normalized_df.drop_duplicates()
)


duplicates_removed = (
    before_duplicates
    - len(normalized_df)
)


print(
    f"\nDuplicate rows removed: "
    f"{duplicates_removed}"
)


# ============================================================
# 10. CHECK MISSING VALUES
# ============================================================

missing_values = (
    normalized_df.isnull().sum().sum()
)


print(
    f"Missing values: "
    f"{missing_values}"
)


# ============================================================
# 11. SAVE
# ============================================================

normalized_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 12. FINAL INFORMATION
# ============================================================

print("\n==========================================")
print(" NORMALIZATION COMPLETE")
print("==========================================")

print(
    f"\nOriginal rows: "
    f"{len(df)}"
)

print(
    f"Normalized unique rows: "
    f"{len(normalized_df)}"
)

print(
    f"Features: "
    f"{len(feature_names)}"
)

print("\nClass distribution:")

print(
    normalized_df["label"]
    .value_counts()
)

print("\nNormalized dataset saved:")

print(OUTPUT_PATH)

print("\n✓ Ready for model training")