from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

# ============================================================
# V4 SESSION DIAGNOSTIC
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "gesture_dataset_v2.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "v4_session_diagnostic_results.txt"

GESTURES = ["fist", "one", "two", "three", "open_palm", "pinch"]

FEATURE_COLUMNS = []
for i in range(21):
    FEATURE_COLUMNS.extend([f"x_{i}", f"y_{i}", f"z_{i}"])


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_hand_scale(row):
    """
    Distance from wrist landmark 0 to middle MCP landmark 9.
    Used as a simple estimate of hand size in image coordinates.
    """
    wrist = np.array([
        row["x_0"],
        row["y_0"],
        row["z_0"]
    ], dtype=float)

    middle_mcp = np.array([
        row["x_9"],
        row["y_9"],
        row["z_9"]
    ], dtype=float)

    return np.linalg.norm(middle_mcp - wrist)


def calculate_wrist_position(row):
    return np.sqrt(
        row["x_0"] ** 2 +
        row["y_0"] ** 2
    )


def session_feature_summary(df):
    records = []

    for session_id, group in df.groupby("session_id"):
        label = group["label"].iloc[0]

        scales = group.apply(calculate_hand_scale, axis=1)
        wrist_positions = group.apply(calculate_wrist_position, axis=1)

        records.append({
            "session_id": session_id,
            "label": label,
            "samples": len(group),
            "hand_scale_mean": scales.mean(),
            "hand_scale_std": scales.std(),
            "hand_scale_min": scales.min(),
            "hand_scale_max": scales.max(),
            "wrist_position_mean": wrist_positions.mean(),
            "wrist_position_std": wrist_positions.std(),
            "wrist_x_mean": group["x_0"].mean(),
            "wrist_y_mean": group["y_0"].mean(),
            "wrist_x_std": group["x_0"].std(),
            "wrist_y_std": group["y_0"].std(),
        })

    return pd.DataFrame(records)


def normalized_landmarks(df):
    """
    Convert raw landmarks to wrist-relative coordinates,
    normalized by wrist-to-middle-MCP distance.
    """
    output = []

    for _, row in df.iterrows():

        landmarks = np.array([
            [row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]]
            for i in range(21)
        ], dtype=float)

        wrist = landmarks[0].copy()
        relative = landmarks - wrist

        scale = np.linalg.norm(relative[9])

        if scale < 1e-8:
            scale = 1.0

        relative = relative / scale

        output.append(relative.flatten())

    return np.asarray(output)


def session_centroid_similarity(df, feature_matrix):
    """
    Calculates pairwise Euclidean distances between session centroids
    after standardizing each feature.
    """
    sessions = df["session_id"].unique()

    session_vectors = []

    for session_id in sessions:
        mask = df["session_id"].values == session_id
        centroid = feature_matrix[mask].mean(axis=0)
        session_vectors.append(centroid)

    session_vectors = np.asarray(session_vectors)

    # Standardize across sessions
    mean = session_vectors.mean(axis=0)
    std = session_vectors.std(axis=0)
    std[std < 1e-8] = 1.0

    standardized = (session_vectors - mean) / std

    records = []

    for i, j in combinations(range(len(sessions)), 2):
        distance = np.linalg.norm(
            standardized[i] - standardized[j]
        )

        records.append({
            "session_a": sessions[i],
            "session_b": sessions[j],
            "distance": distance
        })

    return pd.DataFrame(records)


# ============================================================
# LOAD DATASET
# ============================================================

section("V4 SESSION DIAGNOSTIC")

print(f"Dataset: {DATASET_PATH}")

if not DATASET_PATH.exists():
    print("\nERROR: Dataset not found.")
    print("Expected:")
    print(DATASET_PATH)
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Gestures: {df['label'].nunique()}")


# ============================================================
# BASIC VALIDATION
# ============================================================

section("1. DATASET VALIDATION")

missing = df.isnull().sum().sum()

feature_values = df[FEATURE_COLUMNS].to_numpy(dtype=float)

non_finite = np.sum(~np.isfinite(feature_values))

duplicate_features = df[FEATURE_COLUMNS].duplicated().sum()

print(f"Missing values: {missing}")
print(f"Non-finite feature values: {non_finite}")
print(f"Duplicate feature rows: {duplicate_features}")

print("\nGesture distribution:")
print(df["label"].value_counts().sort_index().to_string())

print("\nSession distribution:")
print(
    df.groupby(["session_id", "label"])
      .size()
      .to_string()
)


# ============================================================
# SESSION SUMMARY
# ============================================================

section("2. SESSION CHARACTERISTICS")

session_summary = session_feature_summary(df)

print(
    session_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# HAND SCALE ANALYSIS
# ============================================================

section("3. HAND SCALE ANALYSIS")

print("\nAverage hand scale by gesture:")

gesture_scale = (
    df.assign(hand_scale=df.apply(calculate_hand_scale, axis=1))
      .groupby("label")["hand_scale"]
      .agg(["mean", "std", "min", "max"])
)

print(
    gesture_scale.to_string(
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nHand scale by session:")

print(
    session_summary[
        [
            "session_id",
            "label",
            "hand_scale_mean",
            "hand_scale_std",
            "hand_scale_min",
            "hand_scale_max"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# WRIST POSITION ANALYSIS
# ============================================================

section("4. WRIST POSITION ANALYSIS")

print("\nWrist position by gesture:")

wrist_summary = (
    df.groupby("label")[
        ["x_0", "y_0"]
    ]
    .agg(["mean", "std", "min", "max"])
)

print(
    wrist_summary.to_string(
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nWrist position by session:")

print(
    session_summary[
        [
            "session_id",
            "label",
            "wrist_x_mean",
            "wrist_y_mean",
            "wrist_x_std",
            "wrist_y_std"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# SESSION-TO-SESSION FEATURE VARIATION
# ============================================================

section("5. SESSION-TO-SESSION FEATURE VARIATION")

X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=float)

pairwise = session_centroid_similarity(
    df,
    X_raw
)

pairwise = pairwise.sort_values("distance")

print("\nMost similar session pairs:")

print(
    pairwise.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nMost different session pairs:")

print(
    pairwise.tail(10).sort_values(
        "distance",
        ascending=False
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# SAME-GESTURE SESSION DISTANCES
# ============================================================

section("6. SAME-GESTURE SESSION CONSISTENCY")

session_labels = (
    df.groupby("session_id")["label"]
      .first()
      .to_dict()
)

same_gesture_pairs = []

for _, row_a in pairwise.iterrows():

    label_a = session_labels[row_a["session_a"]]
    label_b = session_labels[row_a["session_b"]]

    if label_a == label_b:

        same_gesture_pairs.append({
            "gesture": label_a,
            "session_a": row_a["session_a"],
            "session_b": row_a["session_b"],
            "distance": row_a["distance"]
        })

same_gesture_df = pd.DataFrame(same_gesture_pairs)

if len(same_gesture_df) > 0:

    print("\nSame-gesture session distances:")

    print(
        same_gesture_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print("\nAverage same-gesture session distance:")

    print(
        same_gesture_df.groupby("gesture")["distance"]
        .agg(["mean", "min", "max"])
        .to_string(
            float_format=lambda x: f"{x:.5f}"
        )
    )


# ============================================================
# DIFFERENT-GESTURE OVERLAP
# ============================================================

section("7. DIFFERENT-GESTURE SESSION OVERLAP")

different_gesture_pairs = []

for _, row_a in pairwise.iterrows():

    label_a = session_labels[row_a["session_a"]]
    label_b = session_labels[row_a["session_b"]]

    if label_a != label_b:

        different_gesture_pairs.append({
            "gesture_a": label_a,
            "gesture_b": label_b,
            "session_a": row_a["session_a"],
            "session_b": row_a["session_b"],
            "distance": row_a["distance"]
        })

different_gesture_df = pd.DataFrame(
    different_gesture_pairs
)

if len(different_gesture_df) > 0:

    print("\nClosest different-gesture session pairs:")

    print(
        different_gesture_df.sort_values(
            "distance"
        ).head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )


# ============================================================
# RAW VS WRIST-RELATIVE SESSION DISTANCE
# ============================================================

section("8. RAW VS WRIST-RELATIVE REPRESENTATION")

X_relative = normalized_landmarks(df)

relative_pairwise = session_centroid_similarity(
    df,
    X_relative
)

raw_same = []
relative_same = []

for _, row in pairwise.iterrows():

    label_a = session_labels[row["session_a"]]
    label_b = session_labels[row["session_b"]]

    if label_a == label_b:
        raw_same.append(row["distance"])

for _, row in relative_pairwise.iterrows():

    label_a = session_labels[row["session_a"]]
    label_b = session_labels[row["session_b"]]

    if label_a == label_b:
        relative_same.append(row["distance"])

if raw_same and relative_same:

    print(
        f"\nAverage same-gesture RAW distance: "
        f"{np.mean(raw_same):.5f}"
    )

    print(
        f"Average same-gesture WRIST-RELATIVE distance: "
        f"{np.mean(relative_same):.5f}"
    )

    raw_mean = np.mean(raw_same)
    relative_mean = np.mean(relative_same)

    if relative_mean < raw_mean:
        improvement = (
            (raw_mean - relative_mean)
            / raw_mean
            * 100
        )

        print(
            f"\nWrist-relative representation reduces "
            f"same-gesture session distance by "
            f"{improvement:.2f}%."
        )

        print(
            "INTERPRETATION: "
            "wrist-relative coordinates appear more "
            "session-invariant than raw coordinates."
        )

    else:

        print(
            "\nINTERPRETATION: "
            "simple wrist-relative normalization does "
            "not sufficiently reduce session variation."
        )


# ============================================================
# GESTURE FEATURE SEPARABILITY
# ============================================================

section("9. GESTURE SEPARABILITY")

gesture_centroids = (
    df.groupby("label")[FEATURE_COLUMNS]
      .mean()
)

# Standardize feature space using global statistics
global_mean = gesture_centroids.mean(axis=0)
global_std = gesture_centroids.std(axis=0)

global_std[global_std < 1e-8] = 1.0

gesture_standardized = (
    gesture_centroids - global_mean
) / global_std

gesture_distances = []

for gesture_a, gesture_b in combinations(
    GESTURES,
    2
):

    if gesture_a not in gesture_standardized.index:
        continue

    if gesture_b not in gesture_standardized.index:
        continue

    distance = np.linalg.norm(
        gesture_standardized.loc[gesture_a]
        - gesture_standardized.loc[gesture_b]
    )

    gesture_distances.append({
        "gesture_a": gesture_a,
        "gesture_b": gesture_b,
        "distance": distance
    })

gesture_distance_df = pd.DataFrame(
    gesture_distances
).sort_values("distance")

print("\nClosest gesture pairs:")

print(
    gesture_distance_df.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nMost separated gesture pairs:")

print(
    gesture_distance_df.tail(10)
    .sort_values(
        "distance",
        ascending=False
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# FEATURE STABILITY
# ============================================================

section("10. FEATURE STABILITY")

feature_std = df[FEATURE_COLUMNS].std()

feature_variance_table = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "std": feature_std.values
})

print("\nMost variable features:")

print(
    feature_variance_table
    .sort_values(
        "std",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nMost stable features:")

print(
    feature_variance_table
    .sort_values("std")
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# PER-LANDMARK VARIATION
# ============================================================

section("11. LANDMARK VARIATION")

landmark_variation = []

for i in range(21):

    x_std = df[f"x_{i}"].std()
    y_std = df[f"y_{i}"].std()
    z_std = df[f"z_{i}"].std()

    total = np.sqrt(
        x_std ** 2 +
        y_std ** 2 +
        z_std ** 2
    )

    landmark_variation.append({
        "landmark": i,
        "x_std": x_std,
        "y_std": y_std,
        "z_std": z_std,
        "combined_std": total
    })

landmark_df = pd.DataFrame(
    landmark_variation
).sort_values(
    "combined_std",
    ascending=False
)

print(
    landmark_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SESSION LABEL CONSISTENCY
# ============================================================

section("12. SESSION LABEL CONSISTENCY")

session_label_counts = (
    df.groupby("session_id")["label"]
      .nunique()
)

bad_sessions = session_label_counts[
    session_label_counts > 1
]

if len(bad_sessions) == 0:

    print(
        "PASS: Every session contains exactly one gesture."
    )

else:

    print(
        "WARNING: Some sessions contain multiple gestures:"
    )

    print(bad_sessions.to_string())


# ============================================================
# FINAL DIAGNOSTIC DECISION
# ============================================================

section("13. V4 DIAGNOSTIC CONCLUSION")

print(
    """
The V4 diagnostic does NOT train or modify a model.

Use the measurements above to determine:

1. Whether raw landmark coordinates contain excessive
   session-dependent position information.

2. Whether hand-size differences are large between sessions.

3. Whether wrist-relative normalization reduces
   same-gesture session variation.

4. Which gesture pairs have the greatest overlap.

5. Which landmarks/features are unstable.

6. Whether the dataset requires a better geometric
   representation before final model training.
"""
)

if same_gesture_df is not None and len(same_gesture_df) > 0:

    avg_same = same_gesture_df["distance"].mean()

    avg_diff = (
        different_gesture_df["distance"].mean()
        if len(different_gesture_df) > 0
        else 0
    )

    print("\nAutomatic interpretation:")

    print(
        f"Average same-gesture session distance: "
        f"{avg_same:.5f}"
    )

    print(
        f"Average different-gesture session distance: "
        f"{avg_diff:.5f}"
    )

    if avg_diff > avg_same:

        print(
            "\nGOOD SIGN:"
            "\nDifferent gestures are, on average, "
            "more separated than sessions of the same gesture."
        )

    else:

        print(
            "\nWARNING:"
            "\nSession variation is comparable to or larger "
            "than gesture variation."
        )

print(
    """
NEXT STEP:
Do NOT retrain V4 yet.

The next model should be designed from these diagnostic
results rather than blindly changing SVM parameters.
"""
)


# ============================================================
# SAVE REPORT
# ============================================================

report_lines = []

# Re-run the important summaries into a compact report.
report_lines.append("V4 SESSION DIAGNOSTIC REPORT")
report_lines.append("=" * 70)
report_lines.append(f"Rows: {len(df)}")
report_lines.append(f"Columns: {len(df.columns)}")
report_lines.append(f"Sessions: {df['session_id'].nunique()}")
report_lines.append(f"Gestures: {df['label'].nunique()}")
report_lines.append(f"Missing values: {missing}")
report_lines.append(f"Non-finite values: {non_finite}")
report_lines.append(f"Duplicate feature rows: {duplicate_features}")

report_lines.append("\nGESTURE DISTRIBUTION")
report_lines.append(df["label"].value_counts().sort_index().to_string())

report_lines.append("\nSESSION DISTRIBUTION")
report_lines.append(
    df.groupby(["session_id", "label"])
      .size()
      .to_string()
)

report_lines.append("\nSESSION CHARACTERISTICS")
report_lines.append(
    session_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

if len(same_gesture_df) > 0:

    report_lines.append(
        "\nSAME-GESTURE SESSION DISTANCES"
    )

    report_lines.append(
        same_gesture_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

if len(different_gesture_df) > 0:

    report_lines.append(
        "\nCLOSEST DIFFERENT-GESTURE SESSIONS"
    )

    report_lines.append(
        different_gesture_df.sort_values(
            "distance"
        ).head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

report_lines.append(
    "\nCLOSEST GESTURE PAIRS"
)

report_lines.append(
    gesture_distance_df.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

report_lines.append(
    "\nMOST VARIABLE LANDMARKS"
)

report_lines.append(
    landmark_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

report_lines.append(
    "\nEND OF V4 DIAGNOSTIC"
)

OUTPUT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("V4 DIAGNOSTIC COMPLETE")
print("=" * 70)
print(f"\nReport saved to:")
print(OUTPUT_PATH)
print("\nNo dataset or trained model was modified.")
