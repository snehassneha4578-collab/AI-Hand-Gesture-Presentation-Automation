import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\\Users\\sneha\\OneDrive\\Desktop\\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")

if not os.path.exists(DATASET):
    DATASET = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation\dataset\gesture_dataset_v9.csv"

df = pd.read_csv(DATASET)

print("=" * 75)
print("V13 FINGER-STATE SEPARABILITY ANALYSIS")
print("=" * 75)

def p(row, i):
    return np.array([
        row[f"x_{i}"],
        row[f"y_{i}"],
        row[f"z_{i}"]
    ], dtype=float)

def dist(a, b):
    return np.linalg.norm(a - b)

wrist = 0

fingers = {
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20)
}

rows = []

for _, r in df.iterrows():

    w = p(r, 0)
    middle_mcp = p(r, 9)

    scale = dist(w, middle_mcp) + 1e-9

    out = {
        "label": r["label"],
        "session_id": r["session_id"]
    }

    # ------------------------------------------------------
    # Finger extension ratios
    # ------------------------------------------------------

    for name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():

        mcp = p(r, mcp_i)
        pip = p(r, pip_i)
        dip = p(r, dip_i)
        tip = p(r, tip_i)

        direct = dist(w, tip)
        base = dist(w, mcp)

        out[f"{name}_ratio"] = direct / (base + 1e-9)

        out[f"{name}_tip_wrist"] = dist(
            tip, w
        ) / scale

        out[f"{name}_tip_mcp"] = dist(
            tip, mcp
        ) / scale

        out[f"{name}_pip_wrist"] = dist(
            pip, w
        ) / scale

    # ------------------------------------------------------
    # Fingertip distances
    # ------------------------------------------------------

    tips = {
        "index": p(r, 8),
        "middle": p(r, 12),
        "ring": p(r, 16),
        "pinky": p(r, 20),
        "thumb": p(r, 4)
    }

    for a, b in [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
        ("index", "ring"),
        ("index", "pinky"),
        ("middle", "pinky"),
        ("thumb", "index"),
        ("thumb", "middle")
    ]:
        out[f"{a}_{b}"] = dist(
            tips[a], tips[b]
        ) / scale

    # ------------------------------------------------------
    # Relative fingertip height
    # ------------------------------------------------------

    for a, b in [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky")
    ]:
        out[f"{a}_{b}_y"] = (
            tips[a][1] - tips[b][1]
        ) / scale

    rows.append(out)

g = pd.DataFrame(rows)

print(f"\nRows: {len(g)}")
print(f"Gestures: {g['label'].nunique()}")
print(f"Sessions: {g['session_id'].nunique()}")

# ==========================================================
# Compare each feature between relevant gesture pairs
# ==========================================================

features = [
    c for c in g.columns
    if c not in ["label", "session_id"]
]

pairs = [
    ("one", "two"),
    ("two", "three"),
    ("three", "open_palm"),
    ("fist", "one"),
    ("fist", "open_palm"),
    ("pinch", "one"),
    ("pinch", "open_palm")
]

results = []

for feature in features:

    values = {}

    for label in g["label"].unique():
        values[label] = g[
            g["label"] == label
        ][feature].values

    for a, b in pairs:

        if a not in values or b not in values:
            continue

        ma = np.mean(values[a])
        mb = np.mean(values[b])

        pooled = np.sqrt(
            (
                np.var(values[a]) +
                np.var(values[b])
            ) / 2
        )

        separation = abs(ma - mb) / (
            pooled + 1e-9
        )

        results.append({
            "feature": feature,
            "pair": f"{a}_vs_{b}",
            "separation": separation,
            "mean_a": ma,
            "mean_b": mb
        })

result_df = pd.DataFrame(results)

# ==========================================================
# Print strongest features for each important pair
# ==========================================================

print("\n" + "=" * 75)
print("BEST FEATURES FOR ONE vs TWO")
print("=" * 75)

print(
    result_df[
        result_df["pair"] == "one_vs_two"
    ]
    .sort_values("separation", ascending=False)
    .head(10)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("BEST FEATURES FOR TWO vs THREE")
print("=" * 75)

print(
    result_df[
        result_df["pair"] == "two_vs_three"
    ]
    .sort_values("separation", ascending=False)
    .head(10)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("BEST FEATURES FOR THREE vs OPEN PALM")
print("=" * 75)

print(
    result_df[
        result_df["pair"] == "three_vs_open_palm"
    ]
    .sort_values("separation", ascending=False)
    .head(10)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("BEST FEATURES FOR FIST vs ONE")
print("=" * 75)

print(
    result_df[
        result_df["pair"] == "fist_vs_one"
    ]
    .sort_values("separation", ascending=False)
    .head(10)
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("BEST FEATURES FOR PINCH")
print("=" * 75)

print(
    result_df[
        result_df["pair"] == "pinch_vs_one"
    ]
    .sort_values("separation", ascending=False)
    .head(10)
    .round(3)
    .to_string(index=False)
)

# ==========================================================
# Save
# ==========================================================

REPORT = os.path.join(
    BASE_DIR,
    "dataset",
    "v13_finger_state_analysis.txt"
)

with open(REPORT, "w", encoding="utf-8") as f:

    f.write(
        "V13 FINGER-STATE SEPARABILITY ANALYSIS\n"
    )
    f.write("=" * 75 + "\n\n")

    for pair in [
        "one_vs_two",
        "two_vs_three",
        "three_vs_open_palm",
        "fist_vs_one",
        "fist_vs_open_palm",
        "pinch_vs_one",
        "pinch_vs_open_palm"
    ]:

        f.write(
            f"\n{pair.upper()}\n"
        )
        f.write("-" * 60 + "\n")

        f.write(
            result_df[
                result_df["pair"] == pair
            ]
            .sort_values(
                "separation",
                ascending=False
            )
            .head(15)
            .round(4)
            .to_string(index=False)
        )

        f.write("\n")

print("\n" + "=" * 75)
print("V13 ANALYSIS COMPLETE")
print("=" * 75)
print(f"\nReport saved to:\n{REPORT}")
