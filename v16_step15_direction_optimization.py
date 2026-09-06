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

def get_points(row):
    return np.array([
        [float(row[f"x_{i}"]), float(row[f"y_{i}"]), float(row[f"z_{i}"])]
        for i in range(21)
    ])

def direction_features(row):
    p = get_points(row)

    p = p - p[0]

    wrist = p[0]
    middle_mcp = p[9]
    index_mcp = p[5]
    pinky_mcp = p[17]

    scale = max(np.linalg.norm(middle_mcp - wrist), 1e-8)

    forward = middle_mcp - wrist
    forward /= max(np.linalg.norm(forward), 1e-8)

    side = pinky_mcp - index_mcp
    side /= max(np.linalg.norm(side), 1e-8)

    normal = np.cross(side, forward)
    normal /= max(np.linalg.norm(normal), 1e-8)

    side = np.cross(forward, normal)
    side /= max(np.linalg.norm(side), 1e-8)

    chains = {
        "thumb":  [1,2,3,4],
        "index":  [5,6,7,8],
        "middle": [9,10,11,12],
        "ring":   [13,14,15,16],
        "pinky":  [17,18,19,20]
    }

    result = {}

    for finger, chain in chains.items():

        values = []

        for a,b in zip(chain[:-1], chain[1:]):

            v = p[b] - p[a]

            v /= max(np.linalg.norm(v), 1e-8)

            values.extend([
                np.dot(v, side),
                np.dot(v, forward),
                np.dot(v, normal)
            ])

        result[finger] = np.array(values)

    return result

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

FC = [direction_features(row) for _,row in cal.iterrows()]
FV = [direction_features(row) for _,row in val.iterrows()]

y_train = cal["label"].values
y_test = val["label"].values

print("="*78)
print("V16 STEP 15 - DIRECTION FEATURE OPTIMIZATION")
print("="*78)

groups = {
    "thumb": ["thumb"],
    "index": ["index"],
    "middle": ["middle"],
    "ring": ["ring"],
    "pinky": ["pinky"],

    "index+middle": ["index","middle"],
    "ring+pinky": ["ring","pinky"],
    "middle+ring+pinky": ["middle","ring","pinky"],
    "index+middle+ring": ["index","middle","ring"],
    "index+middle+ring+pinky": ["index","middle","ring","pinky"],
    "all_fingers": ["thumb","index","middle","ring","pinky"]
}

results = []

print("\nFeature-family results:\n")

for name, selected in groups.items():

    X_train = np.vstack([
        np.concatenate([d[f] for f in selected])
        for d in FC
    ])

    X_test = np.vstack([
        np.concatenate([d[f] for f in selected])
        for d in FV
    ])

    for k in [1,3,5,7,9,11]:

        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=k,
                weights="distance"
            )
        )

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test,pred)

        results.append(
            (acc,name,k,X_train.shape[1])
        )

    best_family = max(
        [r for r in results if r[1] == name],
        key=lambda x:x[0]
    )

    print(
        f"{name:32s} "
        f"BEST={best_family[0]*100:6.2f}% "
        f"k={best_family[2]:2d} "
        f"features={best_family[3]}"
    )

print("\n" + "="*78)
print("TOP 15 CONFIGURATIONS")
print("="*78)

for acc,name,k,count in sorted(results,reverse=True)[:15]:

    print(
        f"{acc*100:6.2f}% | "
        f"{name:32s} | "
        f"k={k:2d} | "
        f"features={count}"
    )

best = max(results)

print("\n" + "="*78)
print("BEST CONFIGURATION")
print("="*78)

print(f"\nFeature family : {best[1]}")
print(f"KNN k          : {best[2]}")
print(f"Features       : {best[3]}")
print(f"Transfer       : {best[0]*100:.2f}%")

# Evaluate best configuration in detail
selected = groups[best[1]]

X_train = np.vstack([
    np.concatenate([d[f] for f in selected])
    for d in FC
])

X_test = np.vstack([
    np.concatenate([d[f] for f in selected])
    for d in FV
])

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=best[2],
        weights="distance"
    )
)

model.fit(X_train,y_train)
pred = model.predict(X_test)

print("\nPer-gesture accuracy:")

for g in ["one","two","three"]:

    mask = y_test == g
    correct = np.sum(pred[mask] == g)

    print(
        f"{g.upper():6s}: "
        f"{correct/np.sum(mask)*100:6.2f}% "
        f"({correct}/{np.sum(mask)})"
    )

print("\nPrediction distribution:")
print(pd.Series(pred).value_counts().to_dict())

print("\n" + "="*78)
print("DECISION")
print("="*78)

if best[0] >= 0.80:
    print("\nEXCELLENT")
    print("Proceed to confidence rejection + temporal stability.")

elif best[0] >= 0.70:
    print("\nVERY PROMISING")
    print("Build a conservative live recognizer next.")

elif best[0] >= 0.60:
    print("\nPROMISING")
    print("The direction representation is useful.")
    print("Next step should focus on rejection and temporal stability.")

else:
    print("\nNEEDS MORE WORK")
    print("Do not connect presentation controls yet.")

print("\nNo new data collected.")
print("No existing data modified.")
print("="*78)

input("\nPress Enter to return to PowerShell...")
