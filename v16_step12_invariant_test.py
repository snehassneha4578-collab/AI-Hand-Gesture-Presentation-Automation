import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path("C:/Users/sneha/OneDrive/Desktop/AI_Hand_Gesture_Presentation")

CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
VAL = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

def get_points(row):
    return np.array([
        [float(row[f"x_{i}"]), float(row[f"y_{i}"]), float(row[f"z_{i}"])]
        for i in range(21)
    ], dtype=float)

def invariant_features(row):
    p = get_points(row)

    # Translation invariance
    p = p - p[0]

    # Hand coordinate system
    wrist = p[0]
    middle_mcp = p[9]
    index_mcp = p[5]
    pinky_mcp = p[17]

    forward = middle_mcp - wrist
    forward /= max(np.linalg.norm(forward), 1e-8)

    side = pinky_mcp - index_mcp
    side /= max(np.linalg.norm(side), 1e-8)

    normal = np.cross(side, forward)
    normal /= max(np.linalg.norm(normal), 1e-8)

    side = np.cross(forward, normal)
    side /= max(np.linalg.norm(side), 1e-8)

    # Rotation-invariant coordinates
    coords = []

    for point in p:
        v = point - wrist
        coords.extend([
            np.dot(v, side),
            np.dot(v, forward),
            np.dot(v, normal)
        ])

    coords = np.array(coords)

    # Scale invariance
    scale = np.linalg.norm(middle_mcp - wrist)
    scale = max(scale, 1e-8)
    coords /= scale

    f = list(coords)

    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]

    # Wrist-to-tip distances
    for i in tips:
        f.append(np.linalg.norm(p[i] - wrist) / scale)

    # Tip-to-tip distances
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            f.append(np.linalg.norm(p[tips[i]] - p[tips[j]]) / scale)

    # Finger segment lengths
    chains = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]
    ]

    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            f.append(np.linalg.norm(p[a] - p[b]) / scale)

    # MCP-to-tip distances
    for tip, mcp in zip(tips, mcps):
        f.append(np.linalg.norm(p[tip] - p[mcp]) / scale)

    return np.array(f)

print("="*72)
print("V16 STEP 12 - INVARIANT REPRESENTATION TEST")
print("="*72)

if not CAL.exists():
    print("\nERROR: Calibration file not found:")
    print(CAL)
    raise SystemExit

if not VAL.exists():
    print("\nERROR: Validation file not found:")
    print(VAL)
    raise SystemExit

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

print(f"\nCalibration samples : {len(cal)}")
print(f"Validation samples  : {len(val)}")

X_train = np.vstack([invariant_features(row) for _, row in cal.iterrows()])
X_test = np.vstack([invariant_features(row) for _, row in val.iterrows()])

y_train = cal["label"].values
y_test = val["label"].values

print(f"Feature count       : {X_train.shape[1]}")

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean"
    )
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\n" + "="*72)
print("RESULT")
print("="*72)

print(f"\nInvariant representation accuracy: {accuracy*100:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_test,
    pred,
    labels=["one","two","three"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    pred,
    labels=["one","two","three"]
))

print("\nPer-gesture accuracy:")

for g in ["one","two","three"]:
    mask = y_test == g
    acc = np.mean(pred[mask] == g)

    print(
        f"{g.upper():6s}: "
        f"{acc*100:.2f}% "
        f"({np.sum(pred[mask] == g)}/{np.sum(mask)})"
    )

print("\nPrediction distribution:")
print(pd.Series(pred).value_counts().to_dict())

print("\n" + "="*72)
print("COMPARISON")
print("="*72)

print("Previous calibration -> validation centroid : 33.33%")
print(f"Invariant representation + KNN              : {accuracy*100:.2f}%")

if accuracy >= 0.80:
    print("\nEXCELLENT")
    print("Representation is strong enough for live-system testing.")

elif accuracy >= 0.65:
    print("\nGOOD")
    print("Major improvement. We can refine the decision/rejection engine.")

elif accuracy >= 0.50:
    print("\nPARTIAL")
    print("Representation improved, but further refinement is needed.")

else:
    print("\nWEAK")
    print("Representation still has significant domain shift.")
    print("Do NOT connect presentation controls yet.")

print("\nNo new data collected.")
print("No existing data modified.")
print("="*72)

input("\nPress Enter to return to PowerShell...")
