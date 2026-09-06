import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")

CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
TEST = BASE / "dataset" / "v16_calibration" / "v16_step9n_fresh_validation.csv"

FEATURES = [
    "y_1","pinky_ratio","x_20","x_12","x_16","y_2",
    "x_8","x_1","x_11","middle_ratio",
    "y_17","y_13","y_18","y_9","middle_pinky",
    "x_17","ring_pinky_x","x_13","x_15","x_18"
]

def make_features(row):

    pts = np.array([
        [float(row[f"x_{i}"]),
         float(row[f"y_{i}"]),
         float(row[f"z_{i}"])]
        for i in range(21)
    ])

    pts -= pts[0]

    scale = np.linalg.norm(pts[9])

    if scale < 1e-8:
        scale = 1.0

    pts /= scale

    f = {}

    for i in range(21):
        f[f"x_{i}"] = pts[i,0]
        f[f"y_{i}"] = pts[i,1]
        f[f"z_{i}"] = pts[i,2]

    tips = {
        "index":8,
        "middle":12,
        "ring":16,
        "pinky":20
    }

    mcps = {
        "index":5,
        "middle":9,
        "ring":13,
        "pinky":17
    }

    for name,idx in tips.items():

        f[f"{name}_ratio"] = (
            np.linalg.norm(pts[idx]) /
            max(np.linalg.norm(pts[mcps[name]]),1e-8)
        )

    pairs = [
        ("index","middle"),
        ("middle","ring"),
        ("ring","pinky"),
        ("index","ring"),
        ("index","pinky"),
        ("middle","pinky")
    ]

    for a,b in pairs:

        ia = tips[a]
        ib = tips[b]

        f[f"{a}_{b}"] = np.linalg.norm(
            pts[ia] - pts[ib]
        )

        f[f"{a}_{b}_x"] = (
            pts[ia,0] - pts[ib,0]
        )

        f[f"{a}_{b}_y"] = (
            pts[ia,1] - pts[ib,1]
        )

    return np.array([f[x] for x in FEATURES])

cal = pd.read_csv(CAL)
test = pd.read_csv(TEST)

Xcal = np.vstack([
    make_features(row)
    for _,row in cal.iterrows()
])

print("="*72)
print("V16 STEP 10 FEATURE CONSISTENCY AUDIT")
print("="*72)

print(f"Calibration rows : {len(cal)}")
print(f"Fresh-test rows  : {len(test)}")
print(f"Features checked : {len(FEATURES)}")

print("\n" + "="*72)
print("CALIBRATION FEATURE STATISTICS")
print("="*72)

for i,f in enumerate(FEATURES):

    values = Xcal[:,i]

    print(
        f"{f:20s} "
        f"mean={values.mean():8.4f} "
        f"std={values.std():8.4f} "
        f"min={values.min():8.4f} "
        f"max={values.max():8.4f}"
    )

print("\n" + "="*72)
print("FRESH TEST FEATURE STATISTICS")
print("="*72)

missing = [f for f in FEATURES if f not in test.columns]

if missing:

    print("MISSING FEATURES:")
    print(missing)

else:

    Xtest = test[FEATURES].astype(float).values

    for i,f in enumerate(FEATURES):

        values = Xtest[:,i]

        print(
            f"{f:20s} "
            f"mean={values.mean():8.4f} "
            f"std={values.std():8.4f} "
            f"min={values.min():8.4f} "
            f"max={values.max():8.4f}"
        )

    print("\n" + "="*72)
    print("CALIBRATION vs TEST DISTRIBUTION SHIFT")
    print("="*72)

    shifts = []

    for i,f in enumerate(FEATURES):

        a = Xcal[:,i]
        b = Xtest[:,i]

        pooled = np.std(np.concatenate([a,b]))

        if pooled < 1e-8:
            shift = 0
        else:
            shift = abs(a.mean()-b.mean()) / pooled

        shifts.append((shift,f,a.mean(),b.mean()))

    for shift,f,cm,tm in sorted(shifts,reverse=True):

        print(
            f"{f:20s} "
            f"shift={shift:6.3f} "
            f"cal={cm:8.4f} "
            f"test={tm:8.4f}"
        )

    print("\n" + "="*72)
    print("PER-GESTURE TEST DISTRIBUTION")
    print("="*72)

    if "label" in test.columns:

        print(test["label"].value_counts())

        print("\nGesture feature means:")

        for gesture in ["one","two","three"]:

            subset = test[test["label"] == gesture]

            print(f"\n{gesture.upper()} ({len(subset)} samples)")

            for i,f in enumerate(FEATURES):

                if len(subset):

                    values = subset[f].astype(float)

                    print(
                        f"{f:20s}: "
                        f"{values.mean():8.4f}"
                    )

print("\n" + "="*72)
print("AUDIT COMPLETE")
print("="*72)

input("Press Enter to return to PowerShell...")
