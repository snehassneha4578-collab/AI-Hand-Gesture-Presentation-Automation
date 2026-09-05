import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")
REPORT = os.path.join(BASE_DIR, "dataset", "v12_feature_diagnostics.txt")

GESTURES = ["fist", "one", "open_palm", "pinch", "three", "two"]


def point(row, i):
    return np.array([
        float(row[f"x_{i}"]),
        float(row[f"y_{i}"]),
        float(row[f"z_{i}"])
    ])


def distance(a, b):
    return np.linalg.norm(a - b)


def angle(a, b, c):
    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom < 1e-9:
        return 0.0

    value = np.dot(ba, bc) / denom
    value = np.clip(value, -1.0, 1.0)

    return np.degrees(np.arccos(value))


def calculate_features(row):

    wrist = point(row, 0)

    thumb_mcp = point(row, 2)
    thumb_ip = point(row, 3)
    thumb_tip = point(row, 4)

    index_mcp = point(row, 5)
    index_pip = point(row, 6)
    index_dip = point(row, 7)
    index_tip = point(row, 8)

    middle_mcp = point(row, 9)
    middle_pip = point(row, 10)
    middle_dip = point(row, 11)
    middle_tip = point(row, 12)

    ring_mcp = point(row, 13)
    ring_pip = point(row, 14)
    ring_dip = point(row, 15)
    ring_tip = point(row, 16)

    pinky_mcp = point(row, 17)
    pinky_pip = point(row, 18)
    pinky_dip = point(row, 19)
    pinky_tip = point(row, 20)

    palm_scale = distance(wrist, middle_mcp) + 1e-9

    features = {}

    # -----------------------------------------------------
    # FINGER LENGTH / EXTENSION RATIOS
    # -----------------------------------------------------

    fingers = {
        "index": (index_mcp, index_pip, index_dip, index_tip),
        "middle": (middle_mcp, middle_pip, middle_dip, middle_tip),
        "ring": (ring_mcp, ring_pip, ring_dip, ring_tip),
        "pinky": (pinky_mcp, pinky_pip, pinky_dip, pinky_tip)
    }

    for name, (_, pip, dip, tip) in fingers.items():

        features[f"{name}_tip_wrist"] = (
            distance(tip, wrist) / palm_scale
        )

        features[f"{name}_pip_angle"] = angle(
            point(row, 5 if name == "index" else
                  9 if name == "middle" else
                  13 if name == "ring" else 17),
            pip,
            dip
        )

        features[f"{name}_dip_angle"] = angle(
            pip,
            dip,
            tip
        )

    # -----------------------------------------------------
    # THUMB
    # -----------------------------------------------------

    features["thumb_tip_wrist"] = (
        distance(thumb_tip, wrist) / palm_scale
    )

    features["thumb_ip_angle"] = angle(
        thumb_mcp,
        thumb_ip,
        thumb_tip
    )

    # -----------------------------------------------------
    # FINGERTIP PAIR DISTANCES
    # -----------------------------------------------------

    tips = {
        "index": index_tip,
        "middle": middle_tip,
        "ring": ring_tip,
        "pinky": pinky_tip,
        "thumb": thumb_tip
    }

    names = list(tips.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a = names[i]
            b = names[j]

            features[f"{a}_{b}_distance"] = (
                distance(tips[a], tips[b]) / palm_scale
            )

    # -----------------------------------------------------
    # FINGER TIP HEIGHT RELATIONSHIPS
    # -----------------------------------------------------

    features["index_middle_y"] = (
        index_tip[1] - middle_tip[1]
    ) / palm_scale

    features["middle_ring_y"] = (
        middle_tip[1] - ring_tip[1]
    ) / palm_scale

    features["ring_pinky_y"] = (
        ring_tip[1] - pinky_tip[1]
    ) / palm_scale

    features["index_ring_y"] = (
        index_tip[1] - ring_tip[1]
    ) / palm_scale

    features["index_pinky_y"] = (
        index_tip[1] - pinky_tip[1]
    ) / palm_scale

    features["middle_pinky_y"] = (
        middle_tip[1] - pinky_tip[1]
    ) / palm_scale

    # -----------------------------------------------------
    # FINGER TIP X RELATIONSHIPS
    # -----------------------------------------------------

    features["index_middle_x"] = (
        index_tip[0] - middle_tip[0]
    ) / palm_scale

    features["middle_ring_x"] = (
        middle_tip[0] - ring_tip[0]
    ) / palm_scale

    features["ring_pinky_x"] = (
        ring_tip[0] - pinky_tip[0]
    ) / palm_scale

    # -----------------------------------------------------
    # PINCH
    # -----------------------------------------------------

    features["thumb_index_pinch"] = (
        distance(thumb_tip, index_tip) / palm_scale
    )

    features["thumb_middle_distance"] = (
        distance(thumb_tip, middle_tip) / palm_scale
    )

    # -----------------------------------------------------
    # PALM GEOMETRY
    # -----------------------------------------------------

    features["palm_width"] = (
        distance(index_mcp, pinky_mcp) / palm_scale
    )

    features["palm_height"] = (
        distance(wrist, middle_mcp) / palm_scale
    )

    return features


print("=" * 75)
print("V12 LANDMARK GEOMETRY DIAGNOSTIC")
print("=" * 75)

df = pd.read_csv(DATASET)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Gestures: {df['label'].nunique()}")

print("\nGesture counts:")
print(df["label"].value_counts().sort_index())

# ---------------------------------------------------------
# Calculate diagnostic features
# ---------------------------------------------------------

print("\nCalculating interpretable geometry features...")

feature_rows = []

for _, row in df.iterrows():

    features = calculate_features(row)
    features["label"] = row["label"]
    features["session_id"] = row["session_id"]

    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

feature_names = [
    c for c in features_df.columns
    if c not in ["label", "session_id"]
]

print(f"Diagnostic features generated: {len(feature_names)}")

# ---------------------------------------------------------
# Global statistics
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("GESTURE FEATURE SEPARATION")
print("=" * 75)

summary_rows = []

for feature in feature_names:

    groups = []

    for gesture in GESTURES:

        values = features_df[
            features_df["label"] == gesture
        ][feature]

        groups.append({
            "gesture": gesture,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max()
        })

    means = [x["mean"] for x in groups]

    separation = max(means) - min(means)

    overall_std = features_df[feature].std()

    score = separation / (overall_std + 1e-9)

    summary_rows.append({
        "feature": feature,
        "separation_score": score,
        "global_std": overall_std
    })

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "separation_score",
    ascending=False
)

print("\nTop 20 most separating features:")

print(
    summary_df.head(20).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Per-gesture means
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("MOST IMPORTANT FEATURE MEANS")
print("=" * 75)

top_features = summary_df.head(20)["feature"].tolist()

mean_table = features_df.groupby(
    "label"
)[top_features].mean()

print(
    mean_table.round(3).to_string()
)

# ---------------------------------------------------------
# Session variability
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("SESSION VARIABILITY")
print("=" * 75)

session_variation = []

for feature in feature_names:

    session_means = (
        features_df
        .groupby(["label", "session_id"])[feature]
        .mean()
    )

    variation = session_means.groupby(
        level=0
    ).std().mean()

    session_variation.append({
        "feature": feature,
        "session_variation": variation
    })

session_df = pd.DataFrame(
    session_variation
).sort_values(
    "session_variation"
)

print("\nMost stable features across sessions:")

print(
    session_df.head(20).to_string(
        index=False
    )
)

print("\nMost session-variable features:")

print(
    session_df.tail(20).sort_values(
        "session_variation",
        ascending=False
    ).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# Finger extension diagnostics
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("FINGER EXTENSION FEATURE ANALYSIS")
print("=" * 75)

extension_features = [
    c for c in feature_names
    if "tip_wrist" in c
]

extension_means = features_df.groupby(
    "label"
)[extension_features].mean()

print(
    extension_means.round(3).to_string()
)

# ---------------------------------------------------------
# Pinch analysis
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PINCH DISTANCE ANALYSIS")
print("=" * 75)

pinch_feature = "thumb_index_pinch"

print(
    features_df.groupby("label")[pinch_feature]
    .agg(["mean", "std", "min", "max"])
    .round(3)
    .to_string()
)

# ---------------------------------------------------------
# Save complete diagnostic report
# ---------------------------------------------------------

with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V12 LANDMARK GEOMETRY DIAGNOSTIC\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Sessions: {df['session_id'].nunique()}\n"
    )

    f.write(
        f"Gestures: {df['label'].nunique()}\n"
    )

    f.write(
        f"Diagnostic features: {len(feature_names)}\n\n"
    )

    f.write(
        "TOP 20 SEPARATING FEATURES\n"
    )

    f.write(
        summary_df.head(20).to_string(
            index=False
        )
    )

    f.write(
        "\n\nTOP FEATURE MEANS\n"
    )

    f.write(
        mean_table.round(3).to_string()
    )

    f.write(
        "\n\nMOST STABLE FEATURES\n"
    )

    f.write(
        session_df.head(20).to_string(
            index=False
        )
    )

    f.write(
        "\n\nMOST SESSION-VARIABLE FEATURES\n"
    )

    f.write(
        session_df.tail(20)
        .sort_values(
            "session_variation",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    f.write(
        "\n\nFINGER EXTENSION FEATURES\n"
    )

    f.write(
        extension_means.round(3).to_string()
    )

    f.write(
        "\n\nPINCH DISTANCE\n"
    )

    f.write(
        features_df.groupby("label")[pinch_feature]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )

# Save generated features for V12 development
FEATURE_DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "gesture_geometry_v12.csv"
)

features_df.to_csv(
    FEATURE_DATASET,
    index=False
)

print("\n" + "=" * 75)
print("V12 DIAGNOSTIC COMPLETE")
print("=" * 75)

print(f"\nDiagnostic report:")
print(REPORT)

print(f"\nGenerated feature dataset:")
print(FEATURE_DATASET)

print("\nNext step will be determined from these measurements.")
