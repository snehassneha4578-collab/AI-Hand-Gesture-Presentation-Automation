import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("C:/Users/sneha/OneDrive/Desktop/AI_Hand_Gesture_Presentation")

CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
VAL = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

def get_points(row):
    return np.array([
        [float(row[f"x_{i}"]), float(row[f"y_{i}"]), float(row[f"z_{i}"])]
        for i in range(21)
    ])

def features(row):
    p = get_points(row)
    p = p - p[0]

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

    scale = max(np.linalg.norm(middle_mcp - wrist), 1e-8)

    out = []

    # Orientation-normalized landmark coordinates
    for point in p:
        v = point - wrist
        out.extend([
            np.dot(v, side) / scale,
            np.dot(v, forward) / scale,
            np.dot(v, normal) / scale
        ])

    tips = [4,8,12,16,20]
    mcps = [2,5,9,13,17]

    names = []

    # Wrist-to-tip distances
    for k,i in enumerate(tips):
        out.append(np.linalg.norm(p[i]-wrist)/scale)
        names.append(f"tip_{k}_wrist")

    # Tip-to-tip distances
    for i in range(5):
        for j in range(i+1,5):
            out.append(np.linalg.norm(p[tips[i]]-p[tips[j]])/scale)
            names.append(f"tip_{i}_tip_{j}")

    # Finger segment lengths
    finger_names = ["thumb","index","middle","ring","pinky"]
    chains = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]
    ]

    for fname,chain in zip(finger_names,chains):
        for s,(a,b) in enumerate(zip(chain[:-1],chain[1:])):
            out.append(np.linalg.norm(p[a]-p[b])/scale)
            names.append(f"{fname}_segment_{s}")

    # MCP-to-tip
    for fname,tip,mcp in zip(finger_names,tips,mcps):
        out.append(np.linalg.norm(p[tip]-p[mcp])/scale)
        names.append(f"{fname}_mcp_tip")

    return np.array(out), names

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

Xc, names = [], None
Xv = []

for _,row in cal.iterrows():
    f,n = features(row)
    Xc.append(f)
    names = n

for _,row in val.iterrows():
    f,_ = features(row)
    Xv.append(f)

Xc = np.array(Xc)
Xv = np.array(Xv)

print("="*78)
print("V16 STEP 13 - FEATURE STABILITY ANALYSIS")
print("="*78)

print(f"\nCalibration samples : {len(Xc)}")
print(f"Validation samples  : {len(Xv)}")
print(f"Total features      : {Xc.shape[1]}")

gestures = ["one","two","three"]

yc = cal["label"].values
yv = val["label"].values

# Compare class centroids between calibration and validation
print("\n" + "="*78)
print("CLASS CENTROID SHIFT")
print("="*78)

for g in gestures:
    a = Xc[yc == g].mean(axis=0)
    b = Xv[yv == g].mean(axis=0)

    shift = np.abs(a-b)

    print(f"\n{g.upper()}")
    print(f"Mean feature shift: {shift.mean():.4f}")
    print(f"Maximum shift    : {shift.max():.4f}")

# Find features that are stable within each gesture
# but shift strongly between calibration and validation
print("\n" + "="*78)
print("MOST IMPORTANT DOMAIN-SHIFT FEATURES")
print("="*78)

records = []

for i in range(Xc.shape[1]):

    domain_shift = 0

    for g in gestures:
        a = Xc[yc == g, i].mean()
        b = Xv[yv == g, i].mean()

        cal_std = Xc[yc == g, i].std()
        val_std = Xv[yv == g, i].std()

        pooled = np.sqrt((cal_std**2 + val_std**2)/2) + 1e-8

        domain_shift += abs(a-b)/pooled

    records.append((domain_shift, i))

for score,i in sorted(records, reverse=True)[:20]:
    print(
        f"Feature {i:3d} | "
        f"shift={score:6.2f} | "
        f"{names[i] if i < len(names) else 'orientation_coord'}"
    )

# Find features that separate ONE/TWO/THREE in calibration
print("\n" + "="*78)
print("BEST GESTURE-SEPARATING FEATURES")
print("="*78)

separation = []

for i in range(Xc.shape[1]):

    means = [
        Xc[yc == g,i].mean()
        for g in gestures
    ]

    within = np.mean([
        Xc[yc == g,i].std()
        for g in gestures
    ]) + 1e-8

    between = np.std(means)

    score = between / within

    separation.append((score,i,means))

for score,i,means in sorted(separation, reverse=True)[:20]:

    print(
        f"Feature {i:3d} | "
        f"sep={score:6.2f} | "
        f"ONE={means[0]:7.3f} "
        f"TWO={means[1]:7.3f} "
        f"THREE={means[2]:7.3f} | "
        f"{names[i] if i < len(names) else 'orientation_coord'}"
    )

# Identify robust features:
# high gesture separation + low calibration/validation shift
print("\n" + "="*78)
print("ROBUST FEATURE CANDIDATES")
print("="*78)

candidates = []

for score,i,means in separation:

    shifts = []

    for g in gestures:
        a = Xc[yc == g,i].mean()
        b = Xv[yv == g,i].mean()

        pooled = np.sqrt(
            (Xc[yc == g,i].std()**2 +
             Xv[yv == g,i].std()**2)/2
        ) + 1e-8

        shifts.append(abs(a-b)/pooled)

    avg_shift = np.mean(shifts)

    # Strong separation and low domain shift
    robust_score = score / (1 + avg_shift)

    candidates.append(
        (robust_score,score,avg_shift,i,means)
    )

for robust,sep,shift,i,means in sorted(candidates,reverse=True)[:20]:

    print(
        f"Feature {i:3d} | "
        f"robust={robust:6.3f} | "
        f"sep={sep:6.2f} | "
        f"shift={shift:6.2f} | "
        f"ONE={means[0]:7.3f} "
        f"TWO={means[1]:7.3f} "
        f"THREE={means[2]:7.3f} | "
        f"{names[i] if i < len(names) else 'orientation_coord'}"
    )

print("\n" + "="*78)
print("DIAGNOSIS")
print("="*78)

print("""
The goal is NOT to maximize training accuracy.

We are looking for features that:
1. Separate ONE/TWO/THREE.
2. Stay stable between calibration and validation.
3. Do not depend heavily on camera orientation or hand pose.

The next recognizer will use only the strongest robust features.
No new data will be collected.
""")

input("\nPress Enter to return to PowerShell...")
