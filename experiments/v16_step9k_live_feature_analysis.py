import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
DATA = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
OUT = BASE / "dataset" / "v16_live_feature_separation_analysis.txt"

print("=" * 72)
print("V16 STEP 9K - LIVE FEATURE SEPARATION ANALYSIS")
print("=" * 72)

df = pd.read_csv(DATA)

landmark_cols = [
    c for c in df.columns
    if c.startswith("x_") or c.startswith("y_") or c.startswith("z_")
]

print(f"Loaded samples : {len(df)}")
print(f"Landmark cols  : {len(landmark_cols)}")
print(f"Gestures       : {df['label'].value_counts().to_dict()}")

def get_points(row):
    pts = np.zeros((21, 3), dtype=np.float32)

    for i in range(21):
        pts[i, 0] = row[f"x_{i}"]
        pts[i, 1] = row[f"y_{i}"]
        pts[i, 2] = row[f"z_{i}"]

    return pts

def make_features(pts):
    pts = pts.copy()

    # Wrist-centered
    pts -= pts[0]

    # Scale normalization
    scale = np.linalg.norm(pts[9])

    if scale < 1e-6:
        scale = 1.0

    pts /= scale

    f = {}

    # Important landmark distances
    fingertips = {
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20
    }

    mcps = {
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17
    }

    pips = {
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18
    }

    for name, idx in fingertips.items():
        f[f"{name}_tip_wrist"] = np.linalg.norm(pts[idx])
        f[f"{name}_tip_mcp"] = np.linalg.norm(pts[idx] - pts[mcps[name]])

    for name, idx in pips.items():
        f[f"{name}_pip_wrist"] = np.linalg.norm(pts[idx])

    # Finger ratios
    for name in fingertips:
        tip = fingertips[name]
        mcp = mcps[name]
        f[f"{name}_ratio"] = (
            np.linalg.norm(pts[tip]) /
            max(np.linalg.norm(pts[mcp]), 1e-6)
        )

    # Fingertip pair distances
    pairs = [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
        ("index", "ring"),
        ("index", "pinky"),
        ("middle", "pinky")
    ]

    for a, b in pairs:
        f[f"{a}_{b}"] = np.linalg.norm(
            pts[fingertips[a]] - pts[fingertips[b]]
        )

    # Relative fingertip positions
    for a, b in pairs:
        ia = fingertips[a]
        ib = fingertips[b]

        f[f"{a}_{b}_x"] = pts[ia, 0] - pts[ib, 0]
        f[f"{a}_{b}_y"] = pts[ia, 1] - pts[ib, 1]

    # Thumb relationships
    for name, idx in fingertips.items():
        f[f"thumb_{name}"] = np.linalg.norm(
            pts[4] - pts[idx]
        )

    # Raw normalized landmark coordinates
    for i in range(21):
        f[f"x_{i}"] = pts[i, 0]
        f[f"y_{i}"] = pts[i, 1]
        f[f"z_{i}"] = pts[i, 2]

    return f

records = []

for _, row in df.iterrows():
    pts = get_points(row)
    features = make_features(pts)
    features["label"] = row["label"]
    records.append(features)

features_df = pd.DataFrame(records)

gestures = ["one", "two", "three"]

feature_cols = [
    c for c in features_df.columns
    if c != "label"
]

print(f"Generated features: {len(feature_cols)}")
print()

# ------------------------------------------------------------------
# Feature separation using standardized mean difference
# ------------------------------------------------------------------

def separation(g1, g2, feature):
    a = features_df.loc[
        features_df["label"] == g1, feature
    ].astype(float)

    b = features_df.loc[
        features_df["label"] == g2, feature
    ].astype(float)

    pooled = np.sqrt(
        (a.var() + b.var()) / 2
    )

    if pooled < 1e-9:
        return 0.0

    return abs(a.mean() - b.mean()) / pooled

pairs = [
    ("one", "two"),
    ("two", "three"),
    ("one", "three")
]

report = []

report.append("=" * 72)
report.append("V16 STEP 9K - LIVE FEATURE SEPARATION ANALYSIS")
report.append("=" * 72)
report.append("")
report.append(f"Samples: {len(df)}")
report.append(f"Features analyzed: {len(feature_cols)}")
report.append("")

for g1, g2 in pairs:

    scores = []

    for feature in feature_cols:
        s = separation(g1, g2, feature)
        scores.append((feature, s))

    scores.sort(key=lambda x: x[1], reverse=True)

    report.append("-" * 72)
    report.append(
        f"TOP FEATURES: {g1.upper()} vs {g2.upper()}"
    )
    report.append("-" * 72)

    for feature, score in scores[:20]:

        a = features_df.loc[
            features_df["label"] == g1, feature
        ].mean()

        b = features_df.loc[
            features_df["label"] == g2, feature
        ].mean()

        report.append(
            f"{feature:25s} "
            f"score={score:7.3f} | "
            f"{g1}={a:8.3f} | "
            f"{g2}={b:8.3f}"
        )

    report.append("")

# ------------------------------------------------------------------
# Finger-state summary
# ------------------------------------------------------------------

report.append("=" * 72)
report.append("FINGER GEOMETRY SUMMARY")
report.append("=" * 72)

important = [
    "index_tip_wrist",
    "middle_tip_wrist",
    "ring_tip_wrist",
    "pinky_tip_wrist",
    "index_ratio",
    "middle_ratio",
    "ring_ratio",
    "pinky_ratio",
    "index_middle",
    "middle_ring",
    "ring_pinky",
    "index_ring",
    "index_pinky",
    "middle_pinky",
    "ring_pinky_y",
    "index_middle_y"
]

for feature in important:

    if feature not in features_df.columns:
        continue

    report.append("")
    report.append(feature)

    for g in gestures:

        vals = features_df.loc[
            features_df["label"] == g, feature
        ].astype(float)

        report.append(
            f"  {g:5s}: "
            f"mean={vals.mean():.4f}, "
            f"std={vals.std():.4f}, "
            f"min={vals.min():.4f}, "
            f"max={vals.max():.4f}"
        )

# ------------------------------------------------------------------
# Nearest prototype separability
# ------------------------------------------------------------------

report.append("")
report.append("=" * 72)
report.append("INTERPRETATION")
report.append("=" * 72)
report.append("")
report.append(
    "The purpose of this analysis is to identify which geometric "
    "features distinguish ONE, TWO and THREE under the actual "
    "webcam calibration conditions."
)
report.append("")
report.append(
    "Large separation scores indicate features that may be useful "
    "for a personalized decision engine."
)
report.append("")
report.append(
    "TWO is currently the critical failure class according to "
    "the Step 9J live evaluation."
)

OUT.write_text("\n".join(report), encoding="utf-8")

print()
print("=" * 72)
print("TOP FEATURE SEPARATION RESULTS")
print("=" * 72)

for g1, g2 in pairs:

    scores = []

    for feature in feature_cols:
        scores.append(
            (feature, separation(g1, g2, feature))
        )

    scores.sort(key=lambda x: x[1], reverse=True)

    print()
    print(f"{g1.upper()} vs {g2.upper()}")

    for feature, score in scores[:10]:
        print(
            f"  {feature:25s} {score:.3f}"
        )

print()
print("=" * 72)
print("V16 STEP 9K COMPLETE")
print("=" * 72)
print(f"Report saved to:")
print(OUT)
print("=" * 72)

input("Press Enter to return to PowerShell...")
