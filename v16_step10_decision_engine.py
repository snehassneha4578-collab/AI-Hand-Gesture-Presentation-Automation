import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")

CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
TEST = BASE / "dataset" / "v16_calibration" / "v16_step9n_fresh_validation.csv"

OUT = BASE / "dataset" / "v16_step10_decision_engine_results.txt"
MODEL = BASE / "trained_model" / "v16_step10_decision_engine.pkl"

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

    return np.array(
        [f[x] for x in FEATURES],
        dtype=float
    )

print("="*72)
print("V16 STEP 10 - FIXED DECISION ENGINE")
print("="*72)

cal = pd.read_csv(CAL)
test = pd.read_csv(TEST)

print(f"Calibration samples : {len(cal)}")
print(f"Fresh test samples  : {len(test)}")
print()

# Calibration: raw landmarks -> 20 engineered features
X_train = np.vstack([
    make_features(row)
    for _,row in cal.iterrows()
])

y_train = cal["label"].values

# Step 9N already contains the 20 engineered features
missing = [f for f in FEATURES if f not in test.columns]

if missing:
    print("ERROR: Step 9N file is missing:")
    print(missing)
    input("Press Enter...")
    raise SystemExit

X_test = test[FEATURES].astype(float).values
y_test = test["label"].values

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

print(f"Features used: {X_train.shape[1]}")
print()

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=1.0
)

model.fit(X_train_s,y_train)

pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)

accuracy = accuracy_score(y_test,pred)

print("="*72)
print("FRESH LIVE TEST RESULTS")
print("="*72)

print(f"Accuracy: {accuracy*100:.2f}%")
print()

print(classification_report(
    y_test,
    pred,
    labels=["one","two","three"],
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    pred,
    labels=["one","two","three"]
)

print("CONFUSION MATRIX")
print()
print("              one  two  three")

for label,row in zip(
    ["one","two","three"],
    cm
):
    print(
        f"{label:8s}   "
        f"{row[0]:3d}  "
        f"{row[1]:3d}  "
        f"{row[2]:5d}"
    )

print()

print("="*72)
print("PER-GESTURE ACCURACY")
print("="*72)

for gesture in ["one","two","three"]:

    mask = y_test == gesture

    if mask.sum():

        correct = (
            pred[mask] == gesture
        ).sum()

        print(
            f"{gesture.upper():8s}: "
            f"{correct/mask.sum()*100:6.2f}% "
            f"({correct}/{mask.sum()})"
        )

print()

# Probability margin
sorted_probs = np.sort(proba,axis=1)

margin = (
    sorted_probs[:,-1] -
    sorted_probs[:,-2]
)

print("="*72)
print("CONFIDENCE-REJECTION ANALYSIS")
print("="*72)

for threshold in [
    0.10,0.20,0.30,0.40,0.50,0.60
]:

    accepted = margin >= threshold

    coverage = accepted.mean()

    if accepted.sum():

        accepted_accuracy = (
            pred[accepted] ==
            y_test[accepted]
        ).mean()

    else:

        accepted_accuracy = 0

    print(
        f"Margin >= {threshold:.2f} | "
        f"Coverage={coverage*100:6.2f}% | "
        f"Accuracy={accepted_accuracy*100:6.2f}% | "
        f"Accepted={accepted.sum():3d}"
    )

print()

print("="*72)
print("TWO-SPECIFIC ANALYSIS")
print("="*72)

two = y_test == "two"

if two.sum():

    two_correct = (
        pred[two] == "two"
    ).sum()

    print(f"Actual TWO samples : {two.sum()}")
    print(f"Predicted TWO      : {two_correct}")
    print(
        f"TWO recall         : "
        f"{two_correct/two.sum()*100:.2f}%"
    )

    print()
    print("TWO prediction distribution:")

    values,counts = np.unique(
        pred[two],
        return_counts=True
    )

    for value,count in zip(values,counts):

        print(
            f"  {value.upper():8s}: {count}"
        )

MODEL.parent.mkdir(exist_ok=True)

with open(MODEL,"wb") as f:

    pickle.dump({
        "model":model,
        "scaler":scaler,
        "features":FEATURES
    },f)

report = []

report.append(
    "V16 STEP 10 - FIXED CLASS-SPECIFIC LIVE DECISION ENGINE"
)

report.append("="*72)
report.append(f"Calibration samples: {len(cal)}")
report.append(f"Fresh test samples: {len(test)}")
report.append(f"Features: {len(FEATURES)}")
report.append(f"Accuracy: {accuracy*100:.2f}%")
report.append("")
report.append("CONFUSION MATRIX")
report.append(str(cm))
report.append("")
report.append("CLASSIFICATION REPORT")
report.append(
    classification_report(
        y_test,
        pred,
        labels=["one","two","three"],
        zero_division=0
    )
)

OUT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print()
print("="*72)
print("V16 STEP 10 COMPLETE")
print("="*72)
print(f"Model : {MODEL}")
print(f"Report: {OUT}")
print()
print("DO NOT connect PowerPoint yet.")
print("Send me the complete output.")
print("="*72)

input("Press Enter to return to PowerShell...")
