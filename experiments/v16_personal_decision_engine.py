import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_calibration",
    "v16_personal_calibration.csv"
)

REPORT = os.path.join(
    BASE_DIR,
    "dataset",
    "v16_personal_decision_engine_results.txt"
)


def get_points(row):

    return np.array([
        [
            row[f"x_{i}"],
            row[f"y_{i}"],
            row[f"z_{i}"]
        ]
        for i in range(21)
    ], dtype=float)


def normalize(p):

    p = p - p[0]

    scale = np.linalg.norm(p[9])

    if scale < 1e-9:
        return None

    return p / scale


def extract(row):

    p = normalize(get_points(row))

    if p is None:
        return None

    f = {}

    # Landmark indices
    index_tip = 8
    middle_tip = 12
    ring_tip = 16
    pinky_tip = 20

    index_mcp = 5
    middle_mcp = 9
    ring_mcp = 13
    pinky_mcp = 17

    # Tip -> MCP distances
    f["index_tip_mcp"] = np.linalg.norm(
        p[index_tip] - p[index_mcp]
    )

    f["middle_tip_mcp"] = np.linalg.norm(
        p[middle_tip] - p[middle_mcp]
    )

    f["ring_tip_mcp"] = np.linalg.norm(
        p[ring_tip] - p[ring_mcp]
    )

    f["pinky_tip_mcp"] = np.linalg.norm(
        p[pinky_tip] - p[pinky_mcp]
    )

    # Tip -> wrist
    f["ring_tip_wrist"] = np.linalg.norm(
        p[ring_tip]
    )

    f["pinky_tip_wrist"] = np.linalg.norm(
        p[pinky_tip]
    )

    # Finger ratios
    for name, mcp, tip in [
        ("index", index_mcp, index_tip),
        ("middle", middle_mcp, middle_tip),
        ("ring", ring_mcp, ring_tip),
        ("pinky", pinky_mcp, pinky_tip)
    ]:

        f[name + "_ratio"] = (
            np.linalg.norm(p[tip]) /
            (np.linalg.norm(p[mcp]) + 1e-9)
        )

    # Adjacent fingertip distances
    f["index_middle"] = np.linalg.norm(
        p[index_tip] - p[middle_tip]
    )

    f["middle_ring"] = np.linalg.norm(
        p[middle_tip] - p[ring_tip]
    )

    f["ring_pinky"] = np.linalg.norm(
        p[ring_tip] - p[pinky_tip]
    )

    f["index_ring"] = np.linalg.norm(
        p[index_tip] - p[ring_tip]
    )

    # Vertical relationships
    f["index_middle_y"] = (
        p[index_tip,1] -
        p[middle_tip,1]
    )

    f["ring_pinky_y"] = (
        p[ring_tip,1] -
        p[pinky_tip,1]
    )

    # Horizontal relationship
    f["ring_pinky_x"] = (
        p[ring_tip,0] -
        p[pinky_tip,0]
    )

    return f


df = pd.read_csv(DATASET)

df = df[
    df["label"].isin(
        ["one", "two", "three"]
    )
].reset_index(drop=True)

# ---------------------------------------------------------
# Build features
# ---------------------------------------------------------

rows = []

for _, row in df.iterrows():

    f = extract(row)

    if f is not None:

        f["label"] = row["label"]

        rows.append(f)


data = pd.DataFrame(rows)

print("=" * 70)
print("V16 STEP 7 - PERSONALIZED DECISION ENGINE")
print("=" * 70)

print(
    f"\nTotal samples: {len(data)}"
)

# ---------------------------------------------------------
# Split 70/30
# ---------------------------------------------------------

train = []
test = []

for gesture in ["one", "two", "three"]:

    g = data[
        data["label"] == gesture
    ].reset_index(drop=True)

    train.append(g.iloc[:70])
    test.append(g.iloc[70:])

train = pd.concat(
    train,
    ignore_index=True
)

test = pd.concat(
    test,
    ignore_index=True
)

print(
    f"Training samples: {len(train)}"
)

print(
    f"Validation samples: {len(test)}"
)

# ---------------------------------------------------------
# Build means and standard deviations
# ---------------------------------------------------------

features = [
    "ring_tip_mcp",
    "index_middle",
    "ring_pinky_y",
    "ring_pinky",
    "ring_ratio",
    "pinky_tip_mcp",
    "middle_tip_mcp",
    "index_tip_mcp",
    "ring_pinky_x",
    "index_middle_y"
]

stats = {}

for gesture in ["one", "two", "three"]:

    g = train[
        train["label"] == gesture
    ]

    stats[gesture] = {}

    for feature in features:

        stats[gesture][feature] = (
            g[feature].mean(),
            g[feature].std() + 1e-6
        )

# ---------------------------------------------------------
# Standardized prototype distance
# ---------------------------------------------------------

def distance(row, gesture):

    total = 0.0
    count = 0

    for feature in features:

        mean, std = stats[
            gesture
        ][feature]

        z = (
            row[feature] - mean
        ) / std

        total += z * z

        count += 1

    return np.sqrt(
        total / count
    )


predictions = []
truth = []
margin_values = []

for _, row in test.iterrows():

    distances = {
        gesture: distance(
            row,
            gesture
        )
        for gesture in [
            "one",
            "two",
            "three"
        ]
    }

    ordered = sorted(
        distances.items(),
        key=lambda x: x[1]
    )

    prediction = ordered[0][0]

    margin = (
        ordered[1][1] -
        ordered[0][1]
    )

    predictions.append(
        prediction
    )

    truth.append(
        row["label"]
    )

    margin_values.append(
        margin
    )

truth = np.array(truth)
predictions = np.array(predictions)
margin_values = np.array(
    margin_values
)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

accuracy = np.mean(
    truth == predictions
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "Random baseline: 33.33%"
)

print("\nPER-GESTURE")

for gesture in [
    "one",
    "two",
    "three"
]:

    mask = truth == gesture

    score = np.mean(
        predictions[mask] == gesture
    )

    print(
        f"{gesture:8s}: "
        f"{score * 100:.2f}%"
    )

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

gestures = [
    "one",
    "two",
    "three"
]

cm = np.zeros(
    (3,3),
    dtype=int
)

for a, p in zip(
    truth,
    predictions
):

    cm[
        gestures.index(a),
        gestures.index(p)
    ] += 1


cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_" + g
        for g in gestures
    ],
    columns=[
        "Pred_" + g
        for g in gestures
    ]
)

print("\nCONFUSION MATRIX")

print(
    cm_df.to_string()
)

# ---------------------------------------------------------
# Margin analysis
# ---------------------------------------------------------

print("\nCONFIDENCE MARGIN ANALYSIS")

for threshold in [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00
]:

    accepted = (
        margin_values >= threshold
    )

    if np.sum(accepted) == 0:
        continue

    accepted_accuracy = np.mean(
        truth[accepted] ==
        predictions[accepted]
    )

    coverage = np.mean(
        accepted
    )

    print(
        f"Margin >= {threshold:.2f} | "
        f"coverage={coverage*100:.2f}% | "
        f"accuracy={accepted_accuracy*100:.2f}%"
    )

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V16 STEP 7 - PERSONALIZED DECISION ENGINE\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Accuracy: {accuracy*100:.2f}%\n"
    )

    f.write(
        "Random baseline: 33.33%\n\n"
    )

    f.write(
        "PER-GESTURE\n"
    )

    for gesture in gestures:

        mask = truth == gesture

        score = np.mean(
            predictions[mask] == gesture
        )

        f.write(
            f"{gesture}: "
            f"{score*100:.2f}%\n"
        )

    f.write(
        "\nCONFUSION MATRIX\n"
    )

    f.write(
        cm_df.to_string()
    )

    f.write(
        "\n\nCONFIDENCE MARGIN ANALYSIS\n"
    )

    for threshold in [
        0.10,
        0.25,
        0.50,
        0.75,
        1.00
    ]:

        accepted = (
            margin_values >= threshold
        )

        if np.sum(accepted) == 0:
            continue

        accepted_accuracy = np.mean(
            truth[accepted] ==
            predictions[accepted]
        )

        coverage = np.mean(
            accepted
        )

        f.write(
            f"Margin >= {threshold:.2f} | "
            f"coverage={coverage*100:.2f}% | "
            f"accuracy={accepted_accuracy*100:.2f}%\n"
        )


print("\n" + "=" * 70)
print("V16 STEP 7 COMPLETE")
print("=" * 70)

print(
    f"\nReport saved to:\n{REPORT}"
)
