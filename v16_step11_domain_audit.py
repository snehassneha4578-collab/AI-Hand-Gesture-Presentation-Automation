import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")

CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
VAL = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

def normalize(row):
    pts = np.array([
        [float(row[f"x_{i}"]), float(row[f"y_{i}"]), float(row[f"z_{i}"])]
        for i in range(21)
    ])

    pts -= pts[0]

    scale = np.linalg.norm(pts[9])
    if scale < 1e-8:
        scale = 1.0

    pts /= scale
    return pts

def features(row):
    p = normalize(row)

    f = []

    # All normalized landmarks
    f.extend(p.flatten())

    tips = [8,12,16,20]
    mcps = [5,9,13,17]

    # Finger extension ratios
    for tip,mcp in zip(tips,mcps):
        f.append(
            np.linalg.norm(p[tip]) /
            max(np.linalg.norm(p[mcp]),1e-8)
        )

    # Fingertip pairwise distances
    for i in range(4):
        for j in range(i+1,4):
            f.append(np.linalg.norm(p[tips[i]]-p[tips[j]]))

    # Fingertip x/y relationships
    for i in range(4):
        for j in range(i+1,4):
            f.append(p[tips[i],0]-p[tips[j],0])
            f.append(p[tips[i],1]-p[tips[j],1])

    return np.array(f)

print("="*72)
print("V16 STEP 11 - CALIBRATION vs VALIDATION DOMAIN AUDIT")
print("="*72)

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

print(f"\nCalibration: {len(cal)} samples")
print(f"Validation : {len(val)} samples")

print("\nCalibration labels:")
print(cal["label"].value_counts())

print("\nValidation labels:")
print(val["label"].value_counts())

# Generate identical representation
Xcal = np.vstack([features(r) for _,r in cal.iterrows()])
Xval = np.vstack([features(r) for _,r in val.iterrows()])

print(f"\nFeature dimensions: {Xcal.shape[1]}")

print("\n" + "="*72)
print("CLASS CENTROID DISTANCES")
print("="*72)

gestures = ["one","two","three"]

cal_centroids = {}
val_centroids = {}

for g in gestures:

    c = np.vstack([
        features(r)
        for _,r in cal[cal["label"]==g].iterrows()
    ])

    v = np.vstack([
        features(r)
        for _,r in val[val["label"]==g].iterrows()
    ])

    cal_centroids[g] = c.mean(axis=0)
    val_centroids[g] = v.mean(axis=0)

    print(
        f"{g.upper():6s} | "
        f"CAL n={len(c):3d} | "
        f"VAL n={len(v):3d} | "
        f"centroid shift="
        f"{np.linalg.norm(cal_centroids[g]-val_centroids[g]):.4f}"
    )

print("\n" + "="*72)
print("WITHIN-DATA GESTURE SEPARATION")
print("="*72)

for name,centroids in [
    ("CALIBRATION",cal_centroids),
    ("VALIDATION",val_centroids)
]:

    print(f"\n{name}")

    for i in range(len(gestures)):

        for j in range(i+1,len(gestures)):

            a = gestures[i]
            b = gestures[j]

            d = np.linalg.norm(
                centroids[a]-centroids[b]
            )

            print(
                f"{a.upper():5s} vs {b.upper():5s}: "
                f"{d:.4f}"
            )

print("\n" + "="*72)
print("NEAREST-CENTROID VALIDATION")
print("="*72)

# Simple centroid classifier
correct = 0
results = []

for _,row in val.iterrows():

    x = features(row)
    true = row["label"]

    distances = {
        g: np.linalg.norm(x-centroids)
        for g,centroids in val_centroids.items()
    }

    pred = min(distances,key=distances.get)

    results.append((true,pred))

    if pred == true:
        correct += 1

print(
    f"Validation centroid accuracy: "
    f"{correct/len(val)*100:.2f}%"
)

print("\nConfusion:")

for actual in gestures:

    subset = [
        p for t,p in results
        if t == actual
    ]

    print(
        f"{actual.upper():6s} -> "
        f"{pd.Series(subset).value_counts().to_dict()}"
    )

print("\n" + "="*72)
print("CROSS-DOMAIN NEAREST CALIBRATION CENTROID")
print("="*72)

correct = 0
results = []

for _,row in val.iterrows():

    x = features(row)
    true = row["label"]

    distances = {
        g: np.linalg.norm(
            x-cal_centroids[g]
        )
        for g in gestures
    }

    pred = min(distances,key=distances.get)

    results.append((true,pred))

    if pred == true:
        correct += 1

print(
    f"Calibration -> Validation accuracy: "
    f"{correct/len(val)*100:.2f}%"
)

for actual in gestures:

    subset = [
        p for t,p in results
        if t == actual
    ]

    print(
        f"{actual.upper():6s} -> "
        f"{pd.Series(subset).value_counts().to_dict()}"
    )

print("\n" + "="*72)
print("FEATURE-LEVEL DOMAIN SHIFT")
print("="*72)

cal_mean = Xcal.mean(axis=0)
val_mean = Xval.mean(axis=0)
cal_std = Xcal.std(axis=0)

shift = np.abs(cal_mean-val_mean) / np.maximum(cal_std,1e-8)

top = np.argsort(shift)[::-1][:20]

for idx in top:

    print(
        f"Feature {idx:3d}: "
        f"shift={shift[idx]:6.2f} | "
        f"cal={cal_mean[idx]:8.4f} | "
        f"val={val_mean[idx]:8.4f}"
    )

print("\n" + "="*72)
print("FINAL DIAGNOSIS")
print("="*72)

if correct/len(val) >= 0.80:
    print("GOOD: geometry generalizes well.")
    print("The previous classifier/decision engine is the main issue.")

elif correct/len(val) >= 0.50:
    print("PARTIAL: geometry contains useful information.")
    print("We should build a better decision engine.")

else:
    print("WEAK: strong domain/pose mismatch remains.")
    print("The representation needs improvement before automation.")

print()
print("No data was modified.")
print("No new samples were collected.")
print("="*72)

input("Press Enter to return to PowerShell...")
