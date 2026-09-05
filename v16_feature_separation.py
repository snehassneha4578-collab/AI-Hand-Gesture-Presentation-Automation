import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "v16_calibration", "v16_personal_calibration.csv")
REPORT = os.path.join(BASE_DIR, "dataset", "v16_feature_separation_analysis.txt")

df = pd.read_csv(DATASET)

gestures = ["one", "two", "three"]

df = df[df["label"].isin(gestures)].copy()

print("=" * 70)
print("V16 STEP 6 - ONE/TWO/THREE FEATURE SEPARATION")
print("=" * 70)

print(f"\nSamples: {len(df)}")

def points(row):
    return np.array([
        [row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]]
        for i in range(21)
    ], dtype=float)

def features(row):
    p = points(row)

    # Wrist-centered
    p = p - p[0]

    scale = np.linalg.norm(p[9])

    if scale < 1e-9:
        return None

    p = p / scale

    out = {}

    # Fingertip distances from wrist
    names = {
        8: "index_tip",
        12: "middle_tip",
        16: "ring_tip",
        20: "pinky_tip"
    }

    for idx, name in names.items():
        out[name + "_wrist"] = np.linalg.norm(p[idx])

    # Finger extension ratios
    fingers = {
        "index": (5, 8),
        "middle": (9, 12),
        "ring": (13, 16),
        "pinky": (17, 20)
    }

    for name, (mcp, tip) in fingers.items():
        out[name + "_ratio"] = (
            np.linalg.norm(p[tip]) /
            (np.linalg.norm(p[mcp]) + 1e-9)
        )

    # Fingertip pair distances
    pairs = [
        (8,12,"index_middle"),
        (12,16,"middle_ring"),
        (16,20,"ring_pinky"),
        (8,16,"index_ring"),
        (8,20,"index_pinky"),
        (12,20,"middle_pinky")
    ]

    for a,b,name in pairs:
        out[name] = np.linalg.norm(p[a] - p[b])

    # Relative Y positions
    out["index_middle_y"] = p[8,1] - p[12,1]
    out["middle_ring_y"] = p[12,1] - p[16,1]
    out["ring_pinky_y"] = p[16,1] - p[20,1]

    # Relative X positions
    out["index_middle_x"] = p[8,0] - p[12,0]
    out["middle_ring_x"] = p[12,0] - p[16,0]
    out["ring_pinky_x"] = p[16,0] - p[20,0]

    # Tip-to-MCP distances
    for name,(mcp,tip) in fingers.items():
        out[name + "_tip_mcp"] = np.linalg.norm(
            p[tip] - p[mcp]
        )

    return out

rows = []

for _, row in df.iterrows():

    f = features(row)

    if f is not None:
        f["label"] = row["label"]
        rows.append(f)

feature_df = pd.DataFrame(rows)

feature_names = [
    c for c in feature_df.columns
    if c != "label"
]

print(f"Features analyzed: {len(feature_names)}")

# ---------------------------------------------------------
# Separation score
# ---------------------------------------------------------

scores = []

for feature in feature_names:

    means = {}

    for gesture in gestures:
        values = feature_df[
            feature_df["label"] == gesture
        ][feature]

        means[gesture] = values.mean()

    # Between-class variance
    overall = np.mean(list(means.values()))

    between = np.mean([
        (means[g] - overall) ** 2
        for g in gestures
    ])

    # Within-class variance
    within_values = []

    for gesture in gestures:
        values = feature_df[
            feature_df["label"] == gesture
        ][feature]

        within_values.append(values.var())

    within = np.mean(within_values)

    score = between / (within + 1e-9)

    scores.append(
        (feature, score, means)
    )

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 15 DISCRIMINATIVE FEATURES")
print("-" * 70)

for feature, score, means in scores[:15]:

    print(
        f"{feature:22s} "
        f"score={score:8.4f} | "
        f"one={means['one']:.4f} "
        f"two={means['two']:.4f} "
        f"three={means['three']:.4f}"
    )

# ---------------------------------------------------------
# Pairwise separation
# ---------------------------------------------------------

print("\nPAIRWISE MEAN DIFFERENCES")
print("-" * 70)

pairs = [
    ("one","two"),
    ("two","three"),
    ("one","three")
]

for a,b in pairs:

    pair_scores = []

    for feature in feature_names:

        va = feature_df[
            feature_df["label"] == a
        ][feature]

        vb = feature_df[
            feature_df["label"] == b
        ][feature]

        mean_diff = abs(
            va.mean() - vb.mean()
        )

        pooled = np.sqrt(
            (va.var() + vb.var()) / 2
        )

        separation = mean_diff / (
            pooled + 1e-9
        )

        pair_scores.append(
            (feature, separation)
        )

    pair_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{a.upper()} vs {b.upper()}")

    for feature, separation in pair_scores[:8]:

        print(
            f"{feature:22s}: "
            f"{separation:.4f}"
        )

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with open(REPORT, "w", encoding="utf-8") as f:

    f.write(
        "V16 STEP 6 - ONE/TWO/THREE FEATURE SEPARATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Samples: {len(feature_df)}\n"
    )

    f.write(
        f"Features analyzed: {len(feature_names)}\n\n"
    )

    f.write(
        "TOP 15 DISCRIMINATIVE FEATURES\n"
    )

    f.write("-" * 70 + "\n")

    for feature, score, means in scores[:15]:

        f.write(
            f"{feature:22s} "
            f"score={score:.4f} | "
            f"one={means['one']:.4f} "
            f"two={means['two']:.4f} "
            f"three={means['three']:.4f}\n"
        )

    for a,b in pairs:

        f.write(
            f"\n{a.upper()} vs {b.upper()}\n"
        )

        pair_scores = []

        for feature in feature_names:

            va = feature_df[
                feature_df["label"] == a
            ][feature]

            vb = feature_df[
                feature_df["label"] == b
            ][feature]

            mean_diff = abs(
                va.mean() - vb.mean()
            )

            pooled = np.sqrt(
                (va.var() + vb.var()) / 2
            )

            separation = mean_diff / (
                pooled + 1e-9
            )

            pair_scores.append(
                (feature, separation)
            )

        pair_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for feature, separation in pair_scores[:8]:

            f.write(
                f"{feature:22s}: "
                f"{separation:.4f}\n"
            )

print("\n" + "=" * 70)
print("V16 STEP 6 COMPLETE")
print("=" * 70)

print(f"\nReport saved to:\n{REPORT}")

print("\nSend me the TOP 15 DISCRIMINATIVE FEATURES")
print("and the THREE pairwise sections.")
