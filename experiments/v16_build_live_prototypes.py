import csv
import os
import pickle
import numpy as np

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
CSV_PATH = BASE + r"\dataset\v16_calibration\v16_live_calibration_one_two_three.csv"
OUT_PATH = BASE + r"\trained_model\v16_live_prototypes.pkl"

GESTURES = ["one", "two", "three"]

print("=" * 70)
print("V16 STEP 9H - BUILD LIVE PERSONALIZED PROTOTYPES")
print("=" * 70)

with open(CSV_PATH, newline="") as f:
    rows = list(csv.reader(f))

header = rows[0]
data = rows[1:]

label_idx = header.index("label")

# Extract the 63 landmark values.
feature_indices = [
    i for i, name in enumerate(header)
    if name not in ("timestamp", "label")
]

X = np.array(
    [
        [float(row[i]) for i in feature_indices]
        for row in data
    ],
    dtype=np.float32
)

labels = np.array([row[label_idx] for row in data])

print(f"Loaded samples: {len(X)}")
print(f"Features: {X.shape[1]}")

# ------------------------------------------------------------
# Same normalization used by V16
# ------------------------------------------------------------
def normalize(sample):

    pts = sample.reshape(21, 3).copy()

    # Translation invariance
    pts -= pts[0]

    # Scale invariance
    scale = np.linalg.norm(pts[9])

    if scale < 1e-6:
        scale = 1.0

    pts /= scale

    features = list(pts.flatten())

    fingertips = [4, 8, 12, 16, 20]

    # Wrist -> fingertip distances
    for idx in fingertips:
        features.append(np.linalg.norm(pts[idx]))

    # Fingertip pair distances
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(
                np.linalg.norm(
                    pts[fingertips[i]] - pts[fingertips[j]]
                )
            )

    # Relative fingertip coordinates
    for idx in fingertips:
        features.extend(pts[idx].tolist())

    # Thumb-to-fingertip distances
    for idx in [8, 12, 16, 20]:
        features.append(
            np.linalg.norm(pts[4] - pts[idx])
        )

    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------
# Build prototypes
# ------------------------------------------------------------
prototypes = {}

for gesture in GESTURES:

    gesture_rows = X[labels == gesture]

    transformed = np.vstack(
        [normalize(row) for row in gesture_rows]
    )

    prototypes[gesture] = {
        "mean": transformed.mean(axis=0),
        "std": transformed.std(axis=0) + 0.05,
        "count": len(transformed)
    }

    print(
        f"{gesture.upper():5s}: "
        f"{len(transformed)} samples | "
        f"{transformed.shape[1]} features"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "wb") as f:
    pickle.dump(prototypes, f)

print()
print("=" * 70)
print("PROTOTYPES SAVED")
print(OUT_PATH)
print("=" * 70)

input("Press Enter to return to PowerShell...")
