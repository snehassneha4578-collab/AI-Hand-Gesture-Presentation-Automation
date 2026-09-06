import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

BASE = Path("C:/Users/sneha/OneDrive/Desktop/AI_Hand_Gesture_Presentation")
CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
VAL = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

GESTURES = ["one","two","three"]

def points(row):
    return np.array([
        [float(row[f"x_{i}"]), float(row[f"y_{i}"]), float(row[f"z_{i}"])]
        for i in range(21)
    ])

def make_features(row):
    p = points(row)

    # Wrist-centered
    p = p - p[0]

    wrist = p[0]
    scale = max(np.linalg.norm(p[9] - wrist), 1e-8)

    # Hand coordinate frame
    forward = p[9] - wrist
    forward /= max(np.linalg.norm(forward), 1e-8)

    side = p[17] - p[5]
    side /= max(np.linalg.norm(side), 1e-8)

    normal = np.cross(side, forward)
    normal /= max(np.linalg.norm(normal), 1e-8)

    side = np.cross(forward, normal)
    side /= max(np.linalg.norm(side), 1e-8)

    # Orientation-normalized coordinates
    coords = []

    for point in p:
        v = point - wrist
        coords.extend([
            np.dot(v, side) / scale,
            np.dot(v, forward) / scale,
            np.dot(v, normal) / scale
        ])

    coords = np.array(coords)

    tips = [4,8,12,16,20]
    mcps = [2,5,9,13,17]

    # Distance features
    wrist_tip = []
    tip_pairs = []

    for tip in tips:
        wrist_tip.append(np.linalg.norm(p[tip]-wrist)/scale)

    for i in range(5):
        for j in range(i+1,5):
            tip_pairs.append(
                np.linalg.norm(p[tips[i]]-p[tips[j]])/scale
            )

    # Finger segment ratios
    chains = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]
    ]

    finger_ratios = []

    for chain in chains:
        total = 0
        for a,b in zip(chain[:-1],chain[1:]):
            total += np.linalg.norm(p[a]-p[b])

        base = np.linalg.norm(p[chain[0]]-wrist)

        finger_ratios.append(total/max(base,1e-8))

    # MCP-tip ratios
    mcp_tip = []

    for tip,mcp in zip(tips,mcps):
        mcp_tip.append(
            np.linalg.norm(p[tip]-p[mcp])/scale
        )

    # Finger direction vectors
    directions = []

    for chain in chains:
        for a,b in zip(chain[:-1],chain[1:]):
            v = p[b]-p[a]
            v /= max(np.linalg.norm(v),1e-8)

            directions.extend([
                np.dot(v,side),
                np.dot(v,forward),
                np.dot(v,normal)
            ])

    return {
        "coordinates": coords,
        "wrist_tip": np.array(wrist_tip),
        "tip_pairs": np.array(tip_pairs),
        "finger_ratios": np.array(finger_ratios),
        "mcp_tip": np.array(mcp_tip),
        "directions": np.array(directions)
    }

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

FC = [make_features(row) for _,row in cal.iterrows()]
FV = [make_features(row) for _,row in val.iterrows()]

y_train = cal["label"].values
y_test = val["label"].values

print("="*78)
print("V16 STEP 14 - FEATURE FAMILY TRANSFER TEST")
print("="*78)

families = [
    "wrist_tip",
    "tip_pairs",
    "finger_ratios",
    "mcp_tip",
    "directions",
    "wrist_tip+tip_pairs",
    "finger_ratios+mcp_tip",
    "wrist_tip+finger_ratios+mcp_tip",
    "tip_pairs+finger_ratios+mcp_tip",
    "ALL_GEOMETRIC"
]

results = []

for family in families:

    def combine(d):

        if family == "ALL_GEOMETRIC":
            return np.concatenate([
                d["wrist_tip"],
                d["tip_pairs"],
                d["finger_ratios"],
                d["mcp_tip"],
                d["directions"]
            ])

        parts = family.split("+")
        return np.concatenate([d[x] for x in parts])

    X_train = np.vstack([combine(d) for d in FC])
    X_test = np.vstack([combine(d) for d in FV])

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        )
    )

    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    results.append((acc,family,X_train.shape[1]))

    print(
        f"{family:42s} "
        f"{acc*100:6.2f}% "
        f"features={X_train.shape[1]}"
    )

print("\n" + "="*78)
print("BEST FEATURE FAMILIES")
print("="*78)

for acc,family,count in sorted(results,reverse=True):
    print(
        f"{family:42s} "
        f"{acc*100:6.2f}% "
        f"features={count}"
    )

best = max(results)

print("\n" + "="*78)
print("DIAGNOSIS")
print("="*78)

print(f"\nBest representation : {best[1]}")
print(f"Transfer accuracy  : {best[0]*100:.2f}%")

if best[0] >= 0.80:
    print("\nEXCELLENT")
    print("This feature family transfers strongly.")
elif best[0] >= 0.65:
    print("\nGOOD")
    print("This gives us a promising representation.")
elif best[0] >= 0.50:
    print("\nPARTIAL")
    print("Some domain robustness exists, but refinement is required.")
else:
    print("\nWEAK")
    print("Even the best geometric family has poor cross-session transfer.")

print("\nNo new data collected.")
print("No existing data modified.")
print("="*78)

input("\nPress Enter to return to PowerShell...")
