from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset" / "gesture_dataset_v2.csv"
REPORT = BASE_DIR / "dataset" / "v7_data_analysis.txt"

df = pd.read_csv(DATASET)

features = [f"{a}_{i}" for i in range(21) for a in ["x","y","z"]]

print("=" * 75)
print("V7 DATA / SESSION ANALYSIS")
print("=" * 75)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Sessions: {df['session_id'].nunique()}")

print("\nGESTURE COUNTS")
print(df["label"].value_counts().sort_index())

print("\nSESSION COUNTS")
print(df.groupby(["label","session_id"]).size().to_string())

X = df[features].astype(float).values
hands = X.reshape(-1,21,3)

# ------------------------------------------------------------
# Basic coordinate statistics
# ------------------------------------------------------------

print("\nRAW COORDINATE RANGE")
print(f"X: {X[:,0::3].min():.6f} to {X[:,0::3].max():.6f}")
print(f"Y: {X[:,1::3].min():.6f} to {X[:,1::3].max():.6f}")
print(f"Z: {X[:,2::3].min():.6f} to {X[:,2::3].max():.6f}")

# ------------------------------------------------------------
# Per-session centroid / scale
# ------------------------------------------------------------

rows = []

for session_id, group in df.groupby("session_id"):

    arr = group[features].astype(float).values
    lm = arr.reshape(-1,21,3)

    wrist = lm[:,0,:]

    # wrist -> middle MCP
    scale = np.linalg.norm(
        lm[:,9,:] - wrist,
        axis=1
    )

    rows.append({
        "label": group["label"].iloc[0],
        "session_id": session_id,
        "samples": len(group),
        "wrist_x_mean": wrist[:,0].mean(),
        "wrist_y_mean": wrist[:,1].mean(),
        "wrist_z_mean": wrist[:,2].mean(),
        "hand_scale_mean": scale.mean(),
        "hand_scale_std": scale.std()
    })

session_df = pd.DataFrame(rows)

print("\nSESSION GEOMETRY")
print(session_df.to_string(index=False))

# ------------------------------------------------------------
# Gesture-level statistics
# ------------------------------------------------------------

print("\nGESTURE GEOMETRY")

gesture_stats = (
    session_df
    .groupby("label")
    [["wrist_x_mean",
      "wrist_y_mean",
      "wrist_z_mean",
      "hand_scale_mean"]]
    .agg(["mean","std"])
)

print(gesture_stats.to_string())

# ------------------------------------------------------------
# Session variation
# ------------------------------------------------------------

print("\nSESSION VARIATION")

for label, group in session_df.groupby("label"):

    print(f"\n{label}")

    for column in [
        "wrist_x_mean",
        "wrist_y_mean",
        "wrist_z_mean",
        "hand_scale_mean"
    ]:

        values = group[column].values

        if len(values) > 1:
            print(
                f"{column}: "
                f"mean={values.mean():.5f}, "
                f"std={values.std():.5f}, "
                f"range={values.min():.5f}-{values.max():.5f}"
            )

# ------------------------------------------------------------
# Landmark coordinate correlation with session
# ------------------------------------------------------------

print("\nLANDMARK SESSION VARIATION")

session_codes = (
    df["session_id"]
    .astype("category")
    .cat.codes
    .values
)

correlations = []

for i, feature in enumerate(features):

    values = X[:,i]

    corr = np.corrcoef(
        values,
        session_codes
    )[0,1]

    if np.isfinite(corr):
        correlations.append(
            (feature, abs(corr), corr)
        )

correlations.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 20 features correlated with session identity:")

for feature, absolute, signed in correlations[:20]:

    print(
        f"{feature:>5} "
        f"abs_corr={absolute:.4f} "
        f"corr={signed:.4f}"
    )

# ------------------------------------------------------------
# Gesture centroid distances
# ------------------------------------------------------------

print("\nGESTURE CENTROID DISTANCES")

centroids = {}

for label, group in df.groupby("label"):

    centroids[label] = (
        group[features]
        .astype(float)
        .values
        .mean(axis=0)
    )

for i, a in enumerate(sorted(centroids)):

    for b in sorted(centroids)[i+1:]:

        distance = np.linalg.norm(
            centroids[a] - centroids[b]
        )

        print(
            f"{a:>10} <-> {b:<10}: "
            f"{distance:.5f}"
        )

# ------------------------------------------------------------
# Simple wrist-relative representation
# ------------------------------------------------------------

relative = hands - hands[:,0:1,:]

relative_scale = np.linalg.norm(
    relative[:,9,:],
    axis=1,
    keepdims=True
)

relative_scale[
    relative_scale < 1e-6
] = 1.0

relative = relative / relative_scale[:,None,:]

relative_flat = relative.reshape(
    len(relative),
    -1
)

print("\nWRIST-RELATIVE STATISTICS")

print(
    f"Relative min: {relative_flat.min():.6f}"
)

print(
    f"Relative max: {relative_flat.max():.6f}"
)

print(
    f"Relative mean: {relative_flat.mean():.6f}"
)

print(
    f"Relative std: {relative_flat.std():.6f}"
)

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

with open(REPORT, "w", encoding="utf-8") as f:

    f.write("V7 DATA / SESSION ANALYSIS\n")
    f.write("=" * 75 + "\n")

    f.write(f"Rows: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n")
    f.write(f"Sessions: {df['session_id'].nunique()}\n")

    f.write("\nGESTURE COUNTS\n")
    f.write(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    f.write("\n\nSESSION COUNTS\n")
    f.write(
        df.groupby(["label","session_id"])
        .size()
        .to_string()
    )

    f.write("\n\nSESSION GEOMETRY\n")
    f.write(
        session_df.to_string(index=False)
    )

    f.write("\n\nGESTURE GEOMETRY\n")
    f.write(
        gesture_stats.to_string()
    )

    f.write("\n\nTOP SESSION-CORRELATED FEATURES\n")

    for feature, absolute, signed in correlations[:20]:

        f.write(
            f"{feature}: "
            f"abs_corr={absolute:.4f}, "
            f"corr={signed:.4f}\n"
        )

    f.write("\n\nGESTURE CENTROID DISTANCES\n")

    for i, a in enumerate(sorted(centroids)):

        for b in sorted(centroids)[i+1:]:

            distance = np.linalg.norm(
                centroids[a] - centroids[b]
            )

            f.write(
                f"{a} <-> {b}: "
                f"{distance:.5f}\n"
            )

    f.write("\n\nWRIST-RELATIVE STATISTICS\n")
    f.write(
        f"min={relative_flat.min():.6f}\n"
    )
    f.write(
        f"max={relative_flat.max():.6f}\n"
    )
    f.write(
        f"mean={relative_flat.mean():.6f}\n"
    )
    f.write(
        f"std={relative_flat.std():.6f}\n"
    )

print("\n" + "=" * 75)
print("V7 DATA ANALYSIS COMPLETE")
print("=" * 75)

print(f"Report saved:")
print(REPORT)
