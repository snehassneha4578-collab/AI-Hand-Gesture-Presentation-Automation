import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_PATH = "dataset/gesture_dataset.csv"
LIVE_PATH = "dataset/live_diagnostic.csv"


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print(" TRAINING vs LIVE FEATURE COMPARISON")
print("==========================================")

print("\nThis program compares:")
print("Original training dataset")
print("        VS")
print("Live diagnostic dataset")

print("\nNo model will be trained.")
print("No model files will be modified.")


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
        f"\nERROR: Live dataset not found:\n"
        f"{LIVE_PATH}"
    )

    raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

training_df = pd.read_csv(
    TRAINING_PATH
)

live_df = pd.read_csv(
    LIVE_PATH
)

print("✓ Training dataset loaded.")
print("✓ Live dataset loaded.")


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
# BASIC INFORMATION
# ============================================================

print("\n==========================================")
print(" DATASET INFORMATION")
print("==========================================")


print("\nTRAINING DATASET")

print(
    f"Rows     : {len(training_df)}"
)

print(
    f"Columns  : {len(training_df.columns)}"
)


print("\nLIVE DATASET")

print(
    f"Rows     : {len(live_df)}"
)

print(
    f"Columns  : {len(live_df.columns)}"
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_training = [

    f
    for f in feature_names
    if f not in training_df.columns

]


missing_live = [

    f
    for f in feature_names
    if f not in live_df.columns

]


if missing_training:

    print(
        "\nERROR: Features missing from training dataset:"
    )

    print(
        missing_training
    )

    raise SystemExit


if missing_live:

    print(
        "\nERROR: Features missing from live dataset:"
    )

    print(
        missing_live
    )

    raise SystemExit


# ============================================================
# CHECK LABELS
# ============================================================

if "label" not in training_df.columns:

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
# FEATURE MATRICES
# ============================================================

X_train = training_df[
    feature_names
]

X_live = live_df[
    feature_names
]


# ============================================================
# MISSING / INFINITE CHECK
# ============================================================

print("\n==========================================")
print(" DATA QUALITY CHECK")
print("==========================================")


training_missing = (
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


training_inf = np.isinf(
    X_train.values
).sum()


live_inf = np.isinf(
    X_live.values
).sum()


print(
    f"\nTraining missing values : "
    f"{training_missing}"
)

print(
    f"Live missing values     : "
    f"{live_missing}"
)

print(
    f"Training infinite       : "
    f"{training_inf}"
)

print(
    f"Live infinite           : "
    f"{live_inf}"
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print("\n==========================================")
print(" GLOBAL FEATURE STATISTICS")
print("==========================================")


train_mean = X_train.mean()
live_mean = X_live.mean()

train_std = X_train.std()
live_std = X_live.std()

train_min = X_train.min()
live_min = X_live.min()

train_max = X_train.max()
live_max = X_live.max()


mean_difference = (
    live_mean - train_mean
)

absolute_mean_difference = (
    mean_difference.abs()
)

std_difference = (
    live_std - train_std
)

absolute_std_difference = (
    std_difference.abs()
)


comparison = pd.DataFrame({

    "Training_Mean": train_mean,

    "Live_Mean": live_mean,

    "Mean_Difference": mean_difference,

    "Absolute_Mean_Difference":
        absolute_mean_difference,

    "Training_Std": train_std,

    "Live_Std": live_std,

    "Std_Difference": std_difference,

    "Absolute_Std_Difference":
        absolute_std_difference,

    "Training_Min": train_min,

    "Live_Min": live_min,

    "Training_Max": train_max,

    "Live_Max": live_max

})


# ============================================================
# TOP FEATURE DIFFERENCES
# ============================================================

print("\n==========================================")
print(" TOP 15 FEATURES BY MEAN DIFFERENCE")
print("==========================================")


top_mean_features = (

    comparison
    .sort_values(
        "Absolute_Mean_Difference",
        ascending=False
    )
    .head(15)

)


for feature, row in top_mean_features.iterrows():

    print(
        f"{feature:5s} | "
        f"Train Mean: {row['Training_Mean']:.6f} | "
        f"Live Mean: {row['Live_Mean']:.6f} | "
        f"Difference: "
        f"{row['Mean_Difference']:+.6f}"
    )


# ============================================================
# TOP STD DIFFERENCES
# ============================================================

print("\n==========================================")
print(" TOP 15 FEATURES BY STD DIFFERENCE")
print("==========================================")


top_std_features = (

    comparison
    .sort_values(
        "Absolute_Std_Difference",
        ascending=False
    )
    .head(15)

)


for feature, row in top_std_features.iterrows():

    print(
        f"{feature:5s} | "
        f"Train Std: {row['Training_Std']:.6f} | "
        f"Live Std: {row['Live_Std']:.6f} | "
        f"Difference: "
        f"{row['Std_Difference']:+.6f}"
    )


# ============================================================
# RANGE COMPARISON
# ============================================================

print("\n==========================================")
print(" RANGE COMPARISON")
print("==========================================")


train_global_min = X_train.min().min()
train_global_max = X_train.max().max()

live_global_min = X_live.min().min()
live_global_max = X_live.max().max()


print(
    f"\nTraining global minimum: "
    f"{train_global_min:.6f}"
)

print(
    f"Live global minimum:     "
    f"{live_global_min:.6f}"
)

print(
    f"\nTraining global maximum: "
    f"{train_global_max:.6f}"
)

print(
    f"Live global maximum:      "
    f"{live_global_max:.6f}"
)


# ============================================================
# GESTURE LIST
# ============================================================

training_gestures = sorted(
    training_df["label"]
    .unique()
)

live_gestures = sorted(
    live_df["label"]
    .unique()
)


print("\n==========================================")
print(" GESTURE COMPARISON")
print("==========================================")


print(
    "\nTraining gestures:"
)

print(
    training_gestures
)


print(
    "\nLive gestures:"
)

print(
    live_gestures
)


# ============================================================
# PER-GESTURE COMPARISON
# ============================================================

common_gestures = sorted(

    set(training_gestures)
    .intersection(
        live_gestures
    )

)


print("\n==========================================")
print(" PER-GESTURE FEATURE SHIFT")
print("==========================================")


for gesture in common_gestures:

    train_gesture = training_df[
        training_df["label"] == gesture
    ][feature_names]


    live_gesture = live_df[
        live_df["label"] == gesture
    ][feature_names]


    train_gesture_mean = (
        train_gesture.mean()
    )


    live_gesture_mean = (
        live_gesture.mean()
    )


    gesture_difference = (

        live_gesture_mean
        - train_gesture_mean

    )


    average_absolute_difference = (

        gesture_difference
        .abs()
        .mean()

    )


    maximum_absolute_difference = (

        gesture_difference
        .abs()
        .max()

    )


    print(
        f"\n{gesture.upper()}"
    )


    print(
        f"Training samples : "
        f"{len(train_gesture)}"
    )


    print(
        f"Live samples     : "
        f"{len(live_gesture)}"
    )


    print(
        f"Average absolute feature "
        f"mean difference : "
        f"{average_absolute_difference:.6f}"
    )


    print(
        f"Maximum feature mean "
        f"difference       : "
        f"{maximum_absolute_difference:.6f}"
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

print("\n==========================================")
print(" DUPLICATE ANALYSIS")
print("==========================================")


training_duplicates = (
    training_df
    .duplicated()
    .sum()
)


live_duplicates = (
    live_df
    .duplicated()
    .sum()
)


print(
    f"\nTraining duplicate rows : "
    f"{training_duplicates}"
)


print(
    f"Live duplicate rows     : "
    f"{live_duplicates}"
)


# ============================================================
# FEATURE DISTRIBUTION SHIFT SCORE
# ============================================================

print("\n==========================================")
print(" DISTRIBUTION SHIFT SUMMARY")
print("==========================================")


average_mean_shift = (

    comparison[
        "Absolute_Mean_Difference"
    ]
    .mean()

)


median_mean_shift = (

    comparison[
        "Absolute_Mean_Difference"
    ]
    .median()

)


maximum_mean_shift = (

    comparison[
        "Absolute_Mean_Difference"
    ]
    .max()

)


average_std_shift = (

    comparison[
        "Absolute_Std_Difference"
    ]
    .mean()

)


print(
    f"\nAverage absolute mean shift : "
    f"{average_mean_shift:.6f}"
)


print(
    f"Median absolute mean shift  : "
    f"{median_mean_shift:.6f}"
)


print(
    f"Maximum absolute mean shift : "
    f"{maximum_mean_shift:.6f}"
)


print(
    f"Average absolute std shift  : "
    f"{average_std_shift:.6f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n==========================================")
print(" INTERPRETATION")
print("==========================================")


if average_mean_shift < 0.01:

    print(
        "\n✓ LOW GLOBAL FEATURE SHIFT"
    )

    print(
        "The live and training feature"
    )

    print(
        "distributions are relatively similar."
    )

    print(
        "\nThe 9.67% live accuracy may therefore"
    )

    print(
        "require investigation of other factors,"
    )

    print(
        "such as landmark ordering, orientation,"
    )

    print(
        "gesture collection quality, or"
    )

    print(
        "class-specific differences."
    )


elif average_mean_shift < 0.03:

    print(
        "\n⚠ MODERATE FEATURE SHIFT"
    )

    print(
        "There is a noticeable difference"
    )

    print(
        "between training and live features."
    )

    print(
        "\nThis can explain part of the"
    )

    print(
        "live-performance degradation."
    )


else:

    print(
        "\n⚠ LARGE FEATURE SHIFT"
    )

    print(
        "The live feature distribution differs"
    )

    print(
        "substantially from the training data."
    )

    print(
        "\nThis strongly suggests a"
    )

    print(
        "training/live domain mismatch."
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

OUTPUT_PATH = (
    "dataset/"
    "training_vs_live_feature_comparison.csv"
)


comparison.to_csv(
    OUTPUT_PATH
)


print("\n==========================================")
print(" RESULTS SAVED")
print("==========================================")


print(
    f"\nSaved detailed feature comparison to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print(" COMPARISON COMPLETE")
print("==========================================")

print(
    "\nNo model was trained."
)

print(
    "No model files were modified."
)

print(
    "\nUse the results above to determine"
)

print(
    "the correct next ML improvement."
)

print(
    "==========================================\n"
)