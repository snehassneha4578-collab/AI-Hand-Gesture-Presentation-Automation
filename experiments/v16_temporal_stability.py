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
    "v16_temporal_stability_results.txt"
)

GESTURES = ["one", "two", "three"]

# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

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

    tips = {
        "index": (5, 8),
        "middle": (9, 12),
        "ring": (13, 16),
        "pinky": (17, 20)
    }

    for name, (mcp, tip) in tips.items():

        f[name + "_tip_mcp"] = np.linalg.norm(
            p[tip] - p[mcp]
        )

        f[name + "_tip_wrist"] = np.linalg.norm(
            p[tip]
        )

        f[name + "_ratio"] = (
            np.linalg.norm(p[tip]) /
            (np.linalg.norm(p[mcp]) + 1e-9)
        )

    f["index_middle"] = np.linalg.norm(
        p[8] - p[12]
    )

    f["middle_ring"] = np.linalg.norm(
        p[12] - p[16]
    )

    f["ring_pinky"] = np.linalg.norm(
        p[16] - p[20]
    )

    f["index_ring"] = np.linalg.norm(
        p[8] - p[16]
    )

    f["index_middle_y"] = (
        p[8,1] - p[12,1]
    )

    f["ring_pinky_y"] = (
        p[16,1] - p[20,1]
    )

    f["ring_pinky_x"] = (
        p[16,0] - p[20,0]
    )

    return f


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATASET)

df = df[
    df["label"].isin(GESTURES)
].reset_index(drop=True)

print("=" * 70)
print("V16 STEP 8 - TEMPORAL STABILITY + REJECTION")
print("=" * 70)

# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

rows = []

for _, row in df.iterrows():

    f = extract(row)

    if f is not None:

        f["label"] = row["label"]

        rows.append(f)

data = pd.DataFrame(rows)

# ---------------------------------------------------------
# 70/30 split
# ---------------------------------------------------------

train_parts = []
test_parts = []

for gesture in GESTURES:

    g = data[
        data["label"] == gesture
    ].reset_index(drop=True)

    train_parts.append(
        g.iloc[:70]
    )

    test_parts.append(
        g.iloc[70:]
    )

train = pd.concat(
    train_parts,
    ignore_index=True
)

test = pd.concat(
    test_parts,
    ignore_index=True
)

print(
    f"\nTraining samples: {len(train)}"
)

print(
    f"Validation samples: {len(test)}"
)

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

FEATURES = [
    "ring_tip_mcp",
    "index_middle",
    "ring_pinky_y",
    "ring_pinky",
    "ring_ratio",
    "pinky_tip_mcp",
    "middle_tip_mcp",
    "index_tip_mcp",
    "ring_pinky_x",
    "index_middle_y",
    "middle_ring",
    "index_ring"
]

# ---------------------------------------------------------
# Standardized prototype
# ---------------------------------------------------------

stats = {}

for gesture in GESTURES:

    g = train[
        train["label"] == gesture
    ]

    stats[gesture] = {}

    for feature in FEATURES:

        stats[gesture][feature] = (
            g[feature].mean(),
            g[feature].std() + 1e-6
        )


def distances(row):

    result = {}

    for gesture in GESTURES:

        total = 0.0

        for feature in FEATURES:

            mean, std = stats[
                gesture
            ][feature]

            z = (
                row[feature] - mean
            ) / std

            total += z * z

        result[gesture] = np.sqrt(
            total / len(FEATURES)
        )

    return result


# ---------------------------------------------------------
# Individual-frame predictions
# ---------------------------------------------------------

predictions = []
truth = []
margins = []

for _, row in test.iterrows():

    d = distances(row)

    ordered = sorted(
        d.items(),
        key=lambda x: x[1]
    )

    prediction = ordered[0][0]

    margin = (
        ordered[1][1] -
        ordered[0][1]
    )

    predictions.append(prediction)
    truth.append(row["label"])
    margins.append(margin)


predictions = np.array(predictions)
truth = np.array(truth)
margins = np.array(margins)

print("\nINDIVIDUAL FRAME RESULT")

frame_accuracy = np.mean(
    predictions == truth
)

print(
    f"Accuracy: "
    f"{frame_accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# Temporal simulation
# ---------------------------------------------------------
#
# We simulate consecutive frames using the validation
# samples in their original order within each gesture.
#
# A gesture is accepted only when:
#
#   1. margin >= threshold
#   2. the same prediction occurs for N consecutive frames
#
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEMPORAL STABILITY TEST")
print("=" * 70)

def temporal_test(
    threshold,
    stable_frames
):

    accepted = []
    actual = []

    history = []

    for pred, true, margin in zip(
        predictions,
        truth,
        margins
    ):

        if margin >= threshold:

            history.append(pred)

        else:

            history = []

        if len(history) >= stable_frames:

            if all(
                h == history[-1]
                for h in history[-stable_frames:]
            ):

                accepted.append(
                    history[-1]
                )

                actual.append(
                    true
                )

                history = []

    if len(accepted) == 0:

        return 0.0, 0.0, 0

    accepted = np.array(accepted)
    actual = np.array(actual)

    accuracy = np.mean(
        accepted == actual
    )

    coverage = len(accepted) / len(truth)

    return accuracy, coverage, len(accepted)


results = []

for threshold in [
    0.25,
    0.50,
    0.75,
    1.00
]:

    for stable_frames in [
        2,
        3,
        4,
        5
    ]:

        accuracy, coverage, count = temporal_test(
            threshold,
            stable_frames
        )

        results.append(
            (
                threshold,
                stable_frames,
                accuracy,
                coverage,
                count
            )
        )

        print(
            f"Margin >= {threshold:.2f} | "
            f"Stable frames = {stable_frames} | "
            f"Accuracy = {accuracy*100:6.2f}% | "
            f"Coverage = {coverage*100:6.2f}% | "
            f"Accepted = {count}"
        )

# ---------------------------------------------------------
# Best configurations
# ---------------------------------------------------------

valid_results = [
    r for r in results
    if r[4] > 0
]

best_accuracy = max(
    valid_results,
    key=lambda x: x[2]
)

best_balanced = max(
    valid_results,
    key=lambda x: x[2] * x[3]
)

print("\n" + "=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)

print(
    f"\nHighest accuracy:"
    f"\nMargin >= {best_accuracy[0]:.2f}"
    f"\nStable frames = {best_accuracy[1]}"
    f"\nAccuracy = {best_accuracy[2]*100:.2f}%"
    f"\nCoverage = {best_accuracy[3]*100:.2f}%"
)

print(
    f"\nBest accuracy × coverage:"
    f"\nMargin >= {best_balanced[0]:.2f}"
    f"\nStable frames = {best_balanced[1]}"
    f"\nAccuracy = {best_balanced[2]*100:.2f}%"
    f"\nCoverage = {best_balanced[3]*100:.2f}%"
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
        "V16 STEP 8 - TEMPORAL STABILITY + REJECTION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Frame accuracy: "
        f"{frame_accuracy*100:.2f}%\n\n"
    )

    f.write(
        "TEMPORAL TEST RESULTS\n"
    )

    for r in results:

        threshold, stable, acc, cov, count = r

        f.write(
            f"Margin >= {threshold:.2f} | "
            f"Stable frames = {stable} | "
            f"Accuracy = {acc*100:.2f}% | "
            f"Coverage = {cov*100:.2f}% | "
            f"Accepted = {count}\n"
        )

    f.write(
        "\nHIGHEST ACCURACY\n"
    )

    f.write(
        f"Margin >= {best_accuracy[0]:.2f}\n"
        f"Stable frames = {best_accuracy[1]}\n"
        f"Accuracy = {best_accuracy[2]*100:.2f}%\n"
        f"Coverage = {best_accuracy[3]*100:.2f}%\n"
    )

    f.write(
        "\nBEST ACCURACY × COVERAGE\n"
    )

    f.write(
        f"Margin >= {best_balanced[0]:.2f}\n"
        f"Stable frames = {best_balanced[1]}\n"
        f"Accuracy = {best_balanced[2]*100:.2f}%\n"
        f"Coverage = {best_balanced[3]*100:.2f}%\n"
    )

print("\n" + "=" * 70)
print("V16 STEP 8 COMPLETE")
print("=" * 70)

print(
    f"\nReport saved to:\n{REPORT}"
)

print(
    "\nSend me the TEMPORAL STABILITY TEST results "
    "and BEST CONFIGURATIONS."
)
