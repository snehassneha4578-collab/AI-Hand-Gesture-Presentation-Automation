import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# AI HAND GESTURE PROJECT - DATASET V2 DIAGNOSTIC
# ============================================================

print()
print("=" * 70)
print("AI HAND GESTURE PROJECT - DATASET V2 DIAGNOSTIC")
print("=" * 70)


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "gesture_dataset_v2.csv"
)


EXPECTED_FEATURES = 63
EXPECTED_COLUMNS = 65

EXPECTED_GESTURES = [
    "fist",
    "one",
    "two",
    "three",
    "open_palm",
    "pinch"
]


# ============================================================
# 2. FILE CHECK
# ============================================================

print()
print("[1] FILE CHECK")
print("-" * 70)


if not DATASET_PATH.exists():

    print("ERROR: Dataset file not found!")

    print()
    print("Expected location:")
    print(DATASET_PATH)

    raise SystemExit


print("Dataset found:")
print(DATASET_PATH)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print()
print("[2] LOADING DATASET")
print("-" * 70)


try:

    df = pd.read_csv(
        DATASET_PATH
    )

except Exception as error:

    print("ERROR: Could not read dataset.")

    print()
    print(error)

    raise SystemExit


print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ============================================================
# 4. COLUMN VALIDATION
# ============================================================

print()
print("[3] COLUMN VALIDATION")
print("-" * 70)


feature_columns = [
    column
    for column in df.columns
    if column.startswith(("x_", "y_", "z_"))
]


print(
    f"Feature columns found : {len(feature_columns)}"
)

print(
    f"Expected features     : {EXPECTED_FEATURES}"
)


print(
    f"Label column present  : "
    f"{'label' in df.columns}"
)


print(
    f"Session ID present    : "
    f"{'session_id' in df.columns}"
)


print(
    f"Total columns expected: {EXPECTED_COLUMNS}"
)

print(
    f"Total columns found   : {len(df.columns)}"
)


# ============================================================
# 5. CHECK FEATURE NAMES
# ============================================================

print()
print("[4] FEATURE NAME VALIDATION")
print("-" * 70)


missing_features = []

expected_feature_names = []


for i in range(21):

    expected_feature_names.append(
        f"x_{i}"
    )

    expected_feature_names.append(
        f"y_{i}"
    )

    expected_feature_names.append(
        f"z_{i}"
    )


for feature in expected_feature_names:

    if feature not in df.columns:

        missing_features.append(
            feature
        )


if len(missing_features) == 0:

    print(
        "PASS - All 63 expected landmark features are present."
    )

else:

    print(
        "FAIL - Missing feature columns:"
    )

    print(
        missing_features
    )


# ============================================================
# 6. GESTURE DISTRIBUTION
# ============================================================

print()
print("[5] GESTURE DISTRIBUTION")
print("-" * 70)


if "label" not in df.columns:

    print(
        "ERROR: label column is missing."
    )

    raise SystemExit


gesture_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)


print()


for gesture in EXPECTED_GESTURES:

    count = gesture_counts.get(
        gesture,
        0
    )

    percentage = (
        count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{gesture:12s}: "
        f"{count:5d} samples "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 7. UNEXPECTED GESTURES
# ============================================================

print()
print("[6] GESTURE LABEL VALIDATION")
print("-" * 70)


actual_gestures = set(
    df["label"].dropna().unique()
)


expected_gestures_set = set(
    EXPECTED_GESTURES
)


unexpected_gestures = (
    actual_gestures
    - expected_gestures_set
)


missing_gestures = (
    expected_gestures_set
    - actual_gestures
)


if len(unexpected_gestures) == 0:

    print(
        "PASS - No unexpected gesture labels."
    )

else:

    print(
        "WARNING - Unexpected gestures:"
    )

    print(
        unexpected_gestures
    )


if len(missing_gestures) == 0:

    print(
        "PASS - All 6 expected gestures are present."
    )

else:

    print(
        "FAIL - Missing gestures:"
    )

    print(
        missing_gestures
    )


# ============================================================
# 8. SESSION DISTRIBUTION
# ============================================================

print()
print("[7] SESSION DISTRIBUTION")
print("-" * 70)


if "session_id" not in df.columns:

    print(
        "ERROR: session_id column is missing."
    )

else:

    session_counts = (
        df
        .groupby(
            ["label", "session_id"]
        )
        .size()
        .reset_index(
            name="samples"
        )
    )


    print()


    print(
        session_counts.to_string(
            index=False
        )
    )


    print()


    unique_sessions = (
        df["session_id"]
        .nunique()
    )


    print(
        f"Unique sessions: "
        f"{unique_sessions}"
    )


# ============================================================
# 9. MISSING VALUES
# ============================================================

print()
print("[8] MISSING VALUES")
print("-" * 70)


missing_total = (
    df.isnull()
    .sum()
    .sum()
)


print(
    f"Total missing values: "
    f"{missing_total}"
)


if missing_total == 0:

    print(
        "PASS - No missing values."
    )

else:

    print(
        "FAIL - Missing values detected."
    )


# ============================================================
# 10. CONVERT FEATURES TO NUMERIC
# ============================================================

print()
print("[9] NUMERIC FEATURE VALIDATION")
print("-" * 70)


numeric_features = (
    df[feature_columns]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)


non_numeric_count = (
    numeric_features
    .isnull()
    .sum()
    .sum()
)


print(
    f"Invalid/non-numeric feature values: "
    f"{non_numeric_count}"
)


if non_numeric_count == 0:

    print(
        "PASS - All feature values are numeric."
    )

else:

    print(
        "FAIL - Non-numeric feature values detected."
    )


# ============================================================
# 11. NON-FINITE VALUES
# ============================================================

print()
print("[10] NON-FINITE VALUES")
print("-" * 70)


feature_array = (
    numeric_features
    .to_numpy(
        dtype=float
    )
)


nonfinite_count = np.sum(
    ~np.isfinite(
        feature_array
    )
)


print(
    f"Non-finite values: "
    f"{nonfinite_count}"
)


if nonfinite_count == 0:

    print(
        "PASS - All feature values are finite."
    )

else:

    print(
        "FAIL - NaN or infinite values detected."
    )


# ============================================================
# 12. DUPLICATE ANALYSIS
# ============================================================

print()
print("[11] DUPLICATE ANALYSIS")
print("-" * 70)


duplicate_features = (
    df.duplicated(
        subset=feature_columns
    )
    .sum()
)


duplicate_complete = (
    df.duplicated()
    .sum()
)


print(
    f"Duplicate feature rows  : "
    f"{duplicate_features}"
)


print(
    f"Duplicate complete rows : "
    f"{duplicate_complete}"
)


if duplicate_features == 0:

    print(
        "PASS - No duplicate feature rows."
    )

else:

    print(
        "INFO - Duplicate feature rows exist."
    )


# ============================================================
# 13. GLOBAL COORDINATE STATISTICS
# ============================================================

print()
print("[12] GLOBAL COORDINATE STATISTICS")
print("-" * 70)


if feature_array.size > 0:

    print(
        f"Minimum : "
        f"{np.nanmin(feature_array):.6f}"
    )

    print(
        f"Maximum : "
        f"{np.nanmax(feature_array):.6f}"
    )

    print(
        f"Mean    : "
        f"{np.nanmean(feature_array):.6f}"
    )

    print(
        f"Std     : "
        f"{np.nanstd(feature_array):.6f}"
    )


# ============================================================
# 14. WRIST POSITION DIVERSITY
# ============================================================

print()
print("[13] WRIST POSITION DIVERSITY")
print("-" * 70)


for column in [
    "x_0",
    "y_0",
    "z_0"
]:

    if column not in df.columns:

        print(
            f"{column}: MISSING"
        )

        continue


    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )


    print()
    print(column)


    print(
        f"  Min  : "
        f"{values.min():.6f}"
    )

    print(
        f"  Max  : "
        f"{values.max():.6f}"
    )

    print(
        f"  Mean : "
        f"{values.mean():.6f}"
    )

    print(
        f"  Std  : "
        f"{values.std():.6f}"
    )


# ============================================================
# 15. WRIST POSITION BY GESTURE
# ============================================================

print()
print("[14] WRIST POSITION BY GESTURE")
print("-" * 70)


if all(
    column in df.columns
    for column in [
        "x_0",
        "y_0",
        "z_0"
    ]
):

    wrist_stats = (
        df
        .groupby("label")[
            [
                "x_0",
                "y_0",
                "z_0"
            ]
        ]
        .agg(
            [
                "min",
                "max",
                "mean",
                "std"
            ]
        )
    )


    print(
        wrist_stats.to_string()
    )


# ============================================================
# 16. PER-GESTURE FEATURE STATISTICS
# ============================================================

print()
print("[15] PER-GESTURE FEATURE STATISTICS")
print("-" * 70)


for gesture in EXPECTED_GESTURES:

    data = numeric_features[
        df["label"] == gesture
    ]


    if len(data) == 0:

        print()
        print(gesture)
        print("  Samples : 0")

        continue


    print()
    print(gesture)


    print(
        f"  Samples : "
        f"{len(data)}"
    )


    print(
        f"  Mean    : "
        f"{data.mean().mean():.6f}"
    )


    print(
        f"  Std     : "
        f"{data.std().mean():.6f}"
    )


    print(
        f"  Min     : "
        f"{data.min().min():.6f}"
    )


    print(
        f"  Max     : "
        f"{data.max().max():.6f}"
    )


# ============================================================
# 17. COORDINATE BOUNDARY CHECK
# ============================================================

print()
print("[16] COORDINATE BOUNDARY CHECK")
print("-" * 70)


below_zero_total = 0

above_one_total = 0


for column in feature_columns:

    values = numeric_features[column]


    below_zero = (
        values < 0
    ).sum()


    above_one = (
        values > 1
    ).sum()


    below_zero_total += (
        below_zero
    )


    above_one_total += (
        above_one
    )


print(
    f"Total values < 0 : "
    f"{below_zero_total}"
)


print(
    f"Total values > 1 : "
    f"{above_one_total}"
)


print()
print(
    "Note: MediaPipe landmark coordinates can occasionally "
    "fall outside 0-1 near image boundaries."
)


# ============================================================
# 18. SESSION / GESTURE VALIDATION
# ============================================================

print()
print("[17] SESSION / GESTURE VALIDATION")
print("-" * 70)


if (
    "session_id" in df.columns
    and "label" in df.columns
):

    validation_table = (
        df
        .groupby(
            [
                "session_id",
                "label"
            ]
        )
        .size()
        .reset_index(
            name="samples"
        )
    )


    print(
        validation_table.to_string(
            index=False
        )
    )


# ============================================================
# 19. DATASET SIZE
# ============================================================

print()
print("[18] DATASET SIZE")
print("-" * 70)


print(
    f"Total rows: "
    f"{len(df)}"
)


print(
    f"Total columns: "
    f"{len(df.columns)}"
)


print(
    f"Number of gestures: "
    f"{df['label'].nunique()}"
)


if "session_id" in df.columns:

    print(
        f"Number of sessions: "
        f"{df['session_id'].nunique()}"
    )


# ============================================================
# 20. FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET ASSESSMENT")
print("=" * 70)


checks = []


# Feature count
checks.append(
    (
        "63 features",
        len(feature_columns)
        == EXPECTED_FEATURES
    )
)


# Label column
checks.append(
    (
        "Label column",
        "label" in df.columns
    )
)


# Session column
checks.append(
    (
        "Session ID",
        "session_id" in df.columns
    )
)


# Missing values
checks.append(
    (
        "No missing values",
        missing_total == 0
    )
)


# Numeric values
checks.append(
    (
        "All features numeric",
        non_numeric_count == 0
    )
)


# Finite values
checks.append(
    (
        "All values finite",
        nonfinite_count == 0
    )
)


# All gestures
checks.append(
    (
        "All 6 gestures",
        len(missing_gestures) == 0
    )
)


# Expected number of rows
checks.append(
    (
        "At least 1,500 samples",
        len(df) >= 1500
    )
)


# ============================================================
# PRINT CHECK RESULTS
# ============================================================

print()


passed_count = 0


for name, passed in checks:

    if passed:

        status = "PASS"

        passed_count += 1

    else:

        status = "FAIL"


    print(
        f"{status:6s} - {name}"
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)


print(
    f"Dataset rows      : {len(df)}"
)


print(
    f"Dataset columns   : {len(df.columns)}"
)


print(
    f"Gestures          : {df['label'].nunique()}"
)


if "session_id" in df.columns:

    print(
        f"Sessions          : "
        f"{df['session_id'].nunique()}"
    )


print(
    f"Features          : "
    f"{len(feature_columns)}"
)


print(
    f"Missing values    : "
    f"{missing_total}"
)


print(
    f"Non-finite values : "
    f"{nonfinite_count}"
)


print(
    f"Duplicate rows    : "
    f"{duplicate_features}"
)


print(
    f"Checks passed     : "
    f"{passed_count}/{len(checks)}"
)


# ============================================================
# 22. DECISION
# ============================================================

print()
print("=" * 70)
print("DECISION")
print("=" * 70)


all_passed = (
    passed_count
    == len(checks)
)


if all_passed:

    print()
    print(
        "SUCCESS!"
    )

    print(
        "V2 DATASET PASSED ALL REQUIRED CHECKS."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Proceed to V2 session-aware ML training."
    )

else:

    print()
    print(
        "WARNING!"
    )

    print(
        "Some dataset checks failed."
    )

    print()
    print(
        "DO NOT TRAIN THE MODEL YET."
    )

    print(
        "Review the failed checks first."
    )


# ============================================================
# END
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print()