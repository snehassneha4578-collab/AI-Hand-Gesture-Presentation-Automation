import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import Counter

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
DATA = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
OUT = BASE / "trained_model" / "v16_specialized_live_engine.pkl"

df = pd.read_csv(DATA)

def get_points(row):
    pts = np.zeros((21, 3), dtype=np.float32)

    for i in range(21):
        pts[i] = [
            row[f"x_{i}"],
            row[f"y_{i}"],
            row[f"z_{i}"]
        ]

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

    # Direct normalized landmark coordinates
    for i in range(21):
        f[f"x_{i}"] = pts[i, 0]
        f[f"y_{i}"] = pts[i, 1]
        f[f"z_{i}"] = pts[i, 2]

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

    # Finger ratios
    for name in fingertips:
        tip = fingertips[name]
        mcp = mcps[name]

        f[f"{name}_ratio"] = (
            np.linalg.norm(pts[tip]) /
            max(np.linalg.norm(pts[mcp]), 1e-6)
        )

    # Fingertip distances
    for name, idx in fingertips.items():
        f[f"{name}_tip_wrist"] = np.linalg.norm(pts[idx])

    # Pairwise fingertip distances
    pairs = [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
        ("index", "ring"),
        ("index", "pinky"),
        ("middle", "pinky")
    ]

    for a, b in pairs:
        ia = fingertips[a]
        ib = fingertips[b]

        f[f"{a}_{b}"] = np.linalg.norm(
            pts[ia] - pts[ib]
        )

        f[f"{a}_{b}_x"] = pts[ia, 0] - pts[ib, 0]
        f[f"{a}_{b}_y"] = pts[ia, 1] - pts[ib, 1]

    # Thumb relationships
    for name, idx in fingertips.items():
        f[f"thumb_{name}"] = np.linalg.norm(
            pts[4] - pts[idx]
        )

    return f

records = []

for _, row in df.iterrows():
    f = make_features(get_points(row))
    f["label"] = row["label"]
    records.append(f)

features = pd.DataFrame(records)

# ---------------------------------------------------------------
# Strong features discovered in Step 9K
# ---------------------------------------------------------------

FEATURES = [
    "y_1",
    "pinky_ratio",
    "x_20",
    "x_12",
    "x_16",
    "y_2",
    "x_8",
    "x_1",
    "x_11",
    "middle_ratio",

    "y_17",
    "y_13",
    "y_18",
    "y_9",
    "middle_pinky",
    "x_17",
    "ring_pinky_x",
    "x_13",
    "x_15",
    "x_18"
]

FEATURES = [f for f in FEATURES if f in features.columns]

X = features[FEATURES].astype(float).values
y = features["label"].values

# ---------------------------------------------------------------
# Prototype statistics
# ---------------------------------------------------------------

stats = {}

for gesture in ["one", "two", "three"]:

    Xg = X[y == gesture]

    mean = Xg.mean(axis=0)
    std = Xg.std(axis=0)

    std[std < 0.02] = 0.02

    stats[gesture] = {
        "mean": mean,
        "std": std,
        "count": len(Xg)
    }

# ---------------------------------------------------------------
# Pairwise feature weighting
# ---------------------------------------------------------------

weights = np.ones(len(FEATURES), dtype=np.float32)

for i, feature in enumerate(FEATURES):

    values = []

    for g1, g2 in [
        ("one", "two"),
        ("two", "three"),
        ("one", "three")
    ]:

        a = features.loc[
            features["label"] == g1, feature
        ].astype(float).values

        b = features.loc[
            features["label"] == g2, feature
        ].astype(float).values

        pooled = np.sqrt(
            (np.var(a) + np.var(b)) / 2
        )

        if pooled > 1e-9:
            values.append(
                abs(np.mean(a) - np.mean(b)) / pooled
            )

    if values:
        weights[i] = np.mean(values)

# Normalize weights
weights = weights / max(weights.mean(), 1e-6)

engine = {
    "features": FEATURES,
    "stats": stats,
    "weights": weights
}

with open(OUT, "wb") as f:
    pickle.dump(engine, f)

print("=" * 72)
print("V16 STEP 9L - SPECIALIZED LIVE DECISION ENGINE")
print("=" * 72)
print(f"Samples       : {len(df)}")
print(f"Features used : {len(FEATURES)}")
print()

print("FEATURE WEIGHTS")
print("-" * 72)

ranked = sorted(
    zip(FEATURES, weights),
    key=lambda x: x[1],
    reverse=True
)

for name, weight in ranked:
    print(f"{name:25s} {weight:.3f}")

print()
print("=" * 72)
print("ENGINE SAVED")
print("=" * 72)
print(OUT)
print("=" * 72)

input("Press Enter to return to PowerShell...")
