import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"

CAL = os.path.join(
    BASE,"dataset","v16_calibration",
    "v16_live_calibration_one_two_three.csv"
)

VAL = os.path.join(
    BASE,"dataset","v16_calibration",
    "v16_balanced_live_validation.csv"
)

REPORT = os.path.join(
    BASE,"dataset",
    "v16_step24_domain_adaptation_results.txt"
)

np.random.seed(42)

print("="*70)
print("STEP 24 — LIVE DOMAIN ADAPTATION EXPERIMENT")
print("="*70)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

cal = pd.read_csv(CAL)
val = pd.read_csv(VAL)

cols = [
    f"{a}_{i}"
    for i in range(21)
    for a in ["x","y","z"]
]

Xcal = cal[cols].values.astype(float)
ycal = cal["label"].values

Xval = val[cols].values.astype(float)
yval = val["label"].values

Xcal = Xcal.reshape(-1,21,3)
Xval = Xval.reshape(-1,21,3)

# ---------------------------------------------------------
# DIRECTION FEATURES
# ---------------------------------------------------------

fingers = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]
]

def direction_features(X):

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

# ---------------------------------------------------------
# CREATE BALANCED ADAPTATION/EVALUATION SPLIT
# ---------------------------------------------------------

adapt_indices = []
eval_indices = []

for label in ["one","two","three"]:

    idx = np.where(yval == label)[0]

    np.random.shuffle(idx)

    adapt_indices.extend(idx[:50])
    eval_indices.extend(idx[50:])

adapt_indices = np.array(adapt_indices)
eval_indices = np.array(eval_indices)

Dadapt = Dval[adapt_indices]
yadapt = yval[adapt_indices]

Deval = Dval[eval_indices]
yeval = yval[eval_indices]

print("")
print("Adaptation samples :",len(Dadapt))
print("Evaluation samples :",len(Deval))

# ---------------------------------------------------------
# METHOD 1
# ORIGINAL CALIBRATION CENTROIDS
# ---------------------------------------------------------

classes = ["one","two","three"]

cal_centroids = {
    label:Dcal[ycal == label].mean(axis=0)
    for label in classes
}

def nearest_centroid(X,centroids):

    predictions=[]

    for sample in X:

        distances = {
            label:np.linalg.norm(
                sample-centroids[label]
            )
            for label in classes
        }

        predictions.append(
            min(distances,key=distances.get)
        )

    return np.array(predictions)

pred_original = nearest_centroid(
    Deval,
    cal_centroids
)

acc_original = np.mean(
    pred_original == yeval
)*100

# ---------------------------------------------------------
# METHOD 2
# LIVE CENTROID ADAPTATION
# ---------------------------------------------------------

adapt_centroids = {}

for label in classes:

    cal_c = cal_centroids[label]

    live_c = Dadapt[yadapt == label].mean(axis=0)

    # 50/50 interpolation
    adapt_centroids[label] = (
        0.5*cal_c +
        0.5*live_c
    )

pred_adapt = nearest_centroid(
    Deval,
    adapt_centroids
)

acc_adapt = np.mean(
    pred_adapt == yeval
)*100

# ---------------------------------------------------------
# METHOD 3
# PURE LIVE ADAPTATION
# ---------------------------------------------------------

live_centroids = {
    label:Dadapt[yadapt == label].mean(axis=0)
    for label in classes
}

pred_live = nearest_centroid(
    Deval,
    live_centroids
)

acc_live = np.mean(
    pred_live == yeval
)*100

# ---------------------------------------------------------
# PER-GESTURE RESULTS
# ---------------------------------------------------------

def print_per_class(name,pred):

    print("")
    print(name)
    print("-"*50)

    for label in classes:

        mask = yeval == label

        correct = np.sum(
            pred[mask] == yeval[mask]
        )

        total = np.sum(mask)

        print(
            f"{label.upper():5s}: "
            f"{correct}/{total} "
            f"= {correct/total*100:.2f}%"
        )

print("")
print("="*70)
print("RESULTS")
print("="*70)

print("")
print(
    f"Original calibration centroids : "
    f"{acc_original:.2f}%"
)

print(
    f"50/50 adapted centroids        : "
    f"{acc_adapt:.2f}%"
)

print(
    f"Pure live centroids            : "
    f"{acc_live:.2f}%"
)

print_per_class(
    "ORIGINAL CALIBRATION",
    pred_original
)

print_per_class(
    "50/50 DOMAIN ADAPTATION",
    pred_adapt
)

print_per_class(
    "PURE LIVE CENTROIDS",
    pred_live
)

# ---------------------------------------------------------
# CONFUSION MATRICES
# ---------------------------------------------------------

def confusion(pred):

    cm = pd.crosstab(
        pd.Series(
            yeval,
            name="True"
        ),
        pd.Series(
            pred,
            name="Predicted"
        ),
        rownames=["True"],
        colnames=["Predicted"]
    )

    return cm

cm_original = confusion(pred_original)
cm_adapt = confusion(pred_adapt)
cm_live = confusion(pred_live)

print("")
print("Original confusion:")
print(cm_original)

print("")
print("Adapted confusion:")
print(cm_adapt)

print("")
print("Live centroid confusion:")
print(cm_live)

# ---------------------------------------------------------
# IMPROVEMENT
# ---------------------------------------------------------

print("")
print("="*70)
print("STEP 24 DIAGNOSIS")
print("="*70)

improvement = acc_adapt - acc_original

print(
    f"Adaptation improvement: "
    f"{improvement:+.2f} percentage points"
)

if acc_adapt >= 80:

    diagnosis = (
        "STRONG: domain adaptation substantially "
        "improves recognition."
    )

elif acc_adapt >= 65:

    diagnosis = (
        "PROMISING: live-domain adaptation helps, "
        "but further robustness work is needed."
    )

elif acc_adapt > acc_original + 5:

    diagnosis = (
        "IMPROVED: adaptation helps, but accuracy "
        "is still insufficient for commands."
    )

else:

    diagnosis = (
        "WEAK: simple centroid adaptation does not "
        "solve the recognition problem."
    )

print(diagnosis)

# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

with open(REPORT,"w",encoding="utf-8") as f:

    f.write(
        "STEP 24 — LIVE DOMAIN ADAPTATION EXPERIMENT\n"
    )
    f.write("="*70+"\n\n")

    f.write(
        f"Adaptation samples: {len(Dadapt)}\n"
    )

    f.write(
        f"Evaluation samples: {len(Deval)}\n\n"
    )

    f.write(
        f"Original calibration centroids: "
        f"{acc_original:.2f}%\n"
    )

    f.write(
        f"50/50 adapted centroids: "
        f"{acc_adapt:.2f}%\n"
    )

    f.write(
        f"Pure live centroids: "
        f"{acc_live:.2f}%\n\n"
    )

    f.write(
        f"Improvement: "
        f"{improvement:+.2f} percentage points\n\n"
    )

    f.write(
        diagnosis+"\n\n"
    )

    f.write(
        "ORIGINAL CONFUSION MATRIX\n"
    )
    f.write(str(cm_original))
    f.write("\n\n")

    f.write(
        "ADAPTED CONFUSION MATRIX\n"
    )
    f.write(str(cm_adapt))
    f.write("\n\n")

    f.write(
        "PURE LIVE CONFUSION MATRIX\n"
    )
    f.write(str(cm_live))
    f.write("\n")

print("")
print("Report saved:")
print(REPORT)
print("="*70)
