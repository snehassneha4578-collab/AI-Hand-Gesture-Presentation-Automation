from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset" / "gesture_dataset_v7.csv"

df = pd.read_csv(DATASET)

feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]

print("=" * 75)
print("V7 DATASET DIAGNOSTIC")
print("=" * 75)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Features: {len(feature_cols)}")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Gestures: {df['label'].nunique()}")

print("\nGESTURE COUNTS")
print(df["label"].value_counts().sort_index())

print("\nSESSION COUNTS")
print(
    df.groupby(["label", "session_id"])
      .size()
      .to_string()
)

print("\nDATA QUALITY")
print(f"Missing values: {df[feature_cols].isna().sum().sum()}")
print(f"Non-finite values: {np.isfinite(df[feature_cols].to_numpy()).all()}")

duplicates = df[feature_cols + ["label"]].duplicated().sum()
print(f"Duplicate feature+label rows: {duplicates}")

print("\nRAW FEATURE RANGE")
X = df[feature_cols].to_numpy(dtype=float)

print(f"Minimum: {X.min():.6f}")
print(f"Maximum: {X.max():.6f}")
print(f"Mean: {X.mean():.6f}")
print(f"Std: {X.std():.6f}")

# ------------------------------------------------------------
# Session centroids
# ------------------------------------------------------------

print("\nSESSION CENTROID DISTANCES")

session_centroids = (
    df.groupby("session_id")[feature_cols]
      .mean()
)

session_labels = (
    df.groupby("session_id")["label"]
      .first()
)

same_distances = []
different_distances = []

sessions = list(session_centroids.index)

for i in range(len(sessions)):
    for j in range(i + 1, len(sessions)):

        a = sessions[i]
        b = sessions[j]

        distance = np.linalg.norm(
            session_centroids.loc[a].values -
            session_centroids.loc[b].values
        )

        if session_labels[a] == session_labels[b]:
            same_distances.append(distance)
        else:
            different_distances.append(distance)

print(
    f"Same-gesture session distance: "
    f"{np.mean(same_distances):.6f}"
)

print(
    f"Different-gesture session distance: "
    f"{np.mean(different_distances):.6f}"
)

if same_distances and different_distances:

    ratio = (
        np.mean(same_distances)
        /
        np.mean(different_distances)
    )

    print(
        f"Same/Different ratio: {ratio:.4f}"
    )

    if ratio < 0.8:
        print(
            "GOOD: Sessions are more consistent within gestures."
        )
    elif ratio < 1.0:
        print(
            "IMPROVED: Session variation is below gesture variation."
        )
    else:
        print(
            "WARNING: Session variation remains high."
        )

# ------------------------------------------------------------
# Gesture centroid distances
# ------------------------------------------------------------

print("\nGESTURE CENTROID DISTANCES")

gesture_centroids = (
    df.groupby("label")[feature_cols]
      .mean()
)

gestures = sorted(gesture_centroids.index)

pairs = []

for i in range(len(gestures)):
    for j in range(i + 1, len(gestures)):

        a = gestures[i]
        b = gestures[j]

        distance = np.linalg.norm(
            gesture_centroids.loc[a].values -
            gesture_centroids.loc[b].values
        )

        pairs.append(
            (a, b, distance)
        )

for a, b, distance in sorted(
    pairs,
    key=lambda x: x[2]
):

    print(
        f"{a:12} <-> {b:12} : {distance:.6f}"
    )

# ------------------------------------------------------------
# Feature/session correlation
# ------------------------------------------------------------

print("\nFEATURE SESSION-CORRELATION CHECK")

session_codes = (
    df["session_id"]
      .astype("category")
      .cat.codes
)

correlations = []

for feature in feature_cols:

    corr = np.corrcoef(
        df[feature].values,
        session_codes.values
    )[0, 1]

    if np.isfinite(corr):

        correlations.append(
            (feature, abs(corr), corr)
        )

print("\nHighest absolute correlations with session ID:")

for feature, abs_corr, corr in sorted(
    correlations,
    key=lambda x: x[1],
    reverse=True
)[:15]:

    print(
        f"{feature:6} "
        f"abs_corr={abs_corr:.4f} "
        f"corr={corr:.4f}"
    )

# ------------------------------------------------------------
# Feature stability
# ------------------------------------------------------------

print("\nMOST VARIABLE FEATURES")

feature_std = (
    df[feature_cols]
    .std()
    .sort_values(ascending=False)
)

print(
    feature_std.head(15).to_string()
)

print("\nMOST STABLE FEATURES")

print(
    feature_std.tail(15).to_string()
)

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("V7 DIAGNOSTIC COMPLETE")
print("=" * 75)

report_path = BASE_DIR / "dataset" / "v7_diagnostic_results.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("V7 diagnostic completed successfully.\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Sessions: {df['session_id'].nunique()}\n")
    f.write(f"Gestures: {df['label'].nunique()}\n")
    f.write(f"Features: {len(feature_cols)}\n")
    f.write(f"Missing values: {df[feature_cols].isna().sum().sum()}\n")
    f.write(f"Duplicate feature+label rows: {duplicates}\n")

print(
    f"\nReport saved to:\n{report_path}"
)
