import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

CAL = os.path.join(
    BASE,
    "dataset",
    "v16_calibration",
    "v16_live_calibration_one_two_three.csv"
)

VAL = os.path.join(
    BASE,
    "dataset",
    "v16_calibration",
    "v16_balanced_live_validation.csv"
)

REPORT = os.path.join(
    BASE,
    "dataset",
    "v16_step23_domain_shift_results.txt"
)

print("="*70)
print("STEP 23 — LIVE vs CALIBRATION DOMAIN-SHIFT ANALYSIS")
print("="*70)

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

print("")
print("Calibration samples:", len(cal))
print("Validation samples :", len(val))

# ---------------------------------------------------------
# RAW LANDMARK FEATURES
# ---------------------------------------------------------

landmark_cols = [
    f"{a}_{i}"
    for i in range(21)
    for a in ["x","y","z"]
]

Xcal = cal[landmark_cols].values.astype(float)
Xval = val[landmark_cols].values.astype(float)

ycal = cal["label"].values
yval = val["label"].values

# ---------------------------------------------------------
# DIRECTION FEATURES
# Same representation used in Step 15+
# ---------------------------------------------------------

fingers = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]
]

def direction_features(X):

    X = X.reshape(-1,21,3)

    output = []

    for sample in X:

        features = []

        for finger in fingers:

            pts = sample[finger]

            for j in range(3):

                v = pts[j+1] - pts[j]

                norm = np.linalg.norm(v) + 1e-8

                features.extend([
                    v[0]/norm,
                    v[1]/norm,
                    v[2]/norm
                ])

        output.append(features)

    return np.asarray(output)

Dcal = direction_features(Xcal)
Dval = direction_features(Xval)

print("")
print("Direction features:", Dcal.shape[1])

# ---------------------------------------------------------
# DOMAIN SHIFT
# ---------------------------------------------------------

cal_mean = Dcal.mean(axis=0)
cal_std = Dcal.std(axis=0) + 1e-8

val_mean = Dval.mean(axis=0)
val_std = Dval.std(axis=0) + 1e-8

mean_shift = np.abs(val_mean - cal_mean)
std_shift = np.abs(val_std - cal_std)

z_shift = mean_shift / cal_std

# ---------------------------------------------------------
# FEATURE NAMES
# ---------------------------------------------------------

feature_names = []

finger_names = [
    "index",
    "middle",
    "ring",
    "pinky",
    "thumb"
]

for finger_name in finger_names:

    for segment in range(1,4):

        feature_names.extend([
            f"{finger_name}_seg{segment}_dx",
            f"{finger_name}_seg{segment}_dy",
            f"{finger_name}_seg{segment}_dz"
        ])

# ---------------------------------------------------------
# CLASS-SPECIFIC SHIFT
# ---------------------------------------------------------

print("")
print("CLASS-SPECIFIC DOMAIN SHIFT")
print("-"*70)

class_results = []

for label in ["one","two","three"]:

    c = Dcal[ycal == label]
    v = Dval[yval == label]

    cmean = c.mean(axis=0)
    vmean = v.mean(axis=0)

    shift = np.abs(vmean-cmean)/(c.std(axis=0)+1e-8)

    avg = np.mean(shift)
    maximum = np.max(shift)

    class_results.append({
        "gesture": label,
        "average_shift_z": avg,
        "maximum_shift_z": maximum
    })

    print(
        f"{label.upper():5s} | "
        f"Average shift = {avg:.3f} SD | "
        f"Maximum shift = {maximum:.3f} SD"
    )

# ---------------------------------------------------------
# TOP SHIFTED FEATURES
# ---------------------------------------------------------

order = np.argsort(z_shift)[::-1]

print("")
print("TOP 20 DOMAIN-SHIFTED FEATURES")
print("-"*70)

top_features = []

for rank, idx in enumerate(order[:20],1):

    name = feature_names[idx]

    item = {
        "rank": rank,
        "feature": name,
        "cal_mean": cal_mean[idx],
        "val_mean": val_mean[idx],
        "cal_std": cal_std[idx],
        "z_shift": z_shift[idx]
    }

    top_features.append(item)

    print(
        f"{rank:2d}. {name:28s} "
        f"Calibration={cal_mean[idx]: .4f} "
        f"Live={val_mean[idx]: .4f} "
        f"Shift={z_shift[idx]:.2f} SD"
    )

# ---------------------------------------------------------
# CLASS CENTROID DISTANCES
# ---------------------------------------------------------

print("")
print("CALIBRATION → LIVE CENTROID DISTANCES")
print("-"*70)

centroids_cal = {}
centroids_val = {}

for label in ["one","two","three"]:

    centroids_cal[label] = Dcal[ycal == label].mean(axis=0)
    centroids_val[label] = Dval[yval == label].mean(axis=0)

for true_label in ["one","two","three"]:

    distances = {}

    for candidate in ["one","two","three"]:

        distances[candidate] = np.linalg.norm(
            centroids_val[true_label] -
            centroids_cal[candidate]
        )

    ordered = sorted(
        distances.items(),
        key=lambda x:x[1]
    )

    print(
        f"{true_label.upper():5s} live centroid -> "
        f"closest calibration class: "
        f"{ordered[0][0].upper()} "
        f"(distance {ordered[0][1]:.4f})"
    )

# ---------------------------------------------------------
# INTRA-LIVE SEPARATION
# ---------------------------------------------------------

print("")
print("LIVE VALIDATION CLASS SEPARATION")
print("-"*70)

for a,b in [
    ("one","two"),
    ("two","three"),
    ("one","three")
]:

    distance = np.linalg.norm(
        centroids_val[a] -
        centroids_val[b]
    )

    print(
        f"{a.upper()} ↔ {b.upper()} : "
        f"{distance:.4f}"
    )

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

average_domain_shift = np.mean(z_shift)
large_shift_count = int(np.sum(z_shift > 2.0))
very_large_shift_count = int(np.sum(z_shift > 3.0))

print("")
print("="*70)
print("STEP 23 SUMMARY")
print("="*70)

print(
    f"Average feature shift : "
    f"{average_domain_shift:.3f} SD"
)

print(
    f"Features >2 SD shift  : "
    f"{large_shift_count}/45"
)

print(
    f"Features >3 SD shift  : "
    f"{very_large_shift_count}/45"
)

if average_domain_shift < 0.75:
    diagnosis = "LOW DOMAIN SHIFT"
elif average_domain_shift < 1.50:
    diagnosis = "MODERATE DOMAIN SHIFT"
elif average_domain_shift < 2.50:
    diagnosis = "HIGH DOMAIN SHIFT"
else:
    diagnosis = "SEVERE DOMAIN SHIFT"

print("")
print("DIAGNOSIS:", diagnosis)

# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

with open(REPORT,"w",encoding="utf-8") as f:

    f.write("STEP 23 — LIVE vs CALIBRATION DOMAIN-SHIFT ANALYSIS\n")
    f.write("="*70+"\n\n")

    f.write(f"Calibration samples: {len(cal)}\n")
    f.write(f"Validation samples : {len(val)}\n")
    f.write(f"Direction features : {Dcal.shape[1]}\n\n")

    f.write(
        f"Average feature shift: "
        f"{average_domain_shift:.4f} SD\n"
    )

    f.write(
        f"Features >2 SD: "
        f"{large_shift_count}/45\n"
    )

    f.write(
        f"Features >3 SD: "
        f"{very_large_shift_count}/45\n\n"
    )

    f.write(f"Diagnosis: {diagnosis}\n\n")

    f.write("CLASS-SPECIFIC SHIFT\n")
    f.write("-"*70+"\n")

    for item in class_results:

        f.write(
            f"{item['gesture'].upper()}: "
            f"average={item['average_shift_z']:.4f} SD, "
            f"maximum={item['maximum_shift_z']:.4f} SD\n"
        )

    f.write("\nTOP 20 SHIFTED FEATURES\n")
    f.write("-"*70+"\n")

    for item in top_features:

        f.write(
            f"{item['rank']:2d}. "
            f"{item['feature']:28s} "
            f"cal={item['cal_mean']:.4f} "
            f"live={item['val_mean']:.4f} "
            f"shift={item['z_shift']:.2f} SD\n"
        )

    f.write("\nLIVE CLASS SEPARATION\n")
    f.write("-"*70+"\n")

    for a,b in [
        ("one","two"),
        ("two","three"),
        ("one","three")
    ]:

        distance = np.linalg.norm(
            centroids_val[a] -
            centroids_val[b]
        )

        f.write(
            f"{a.upper()} <-> {b.upper()}: "
            f"{distance:.4f}\n"
        )

print("")
print("Report saved:")
print(REPORT)
print("="*70)
