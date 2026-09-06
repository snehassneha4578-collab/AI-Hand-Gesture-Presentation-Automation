import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")

CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
TEST = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

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

    # Wrist-centered
    pts = pts - pts[0]

    # Scale normalization
    scale = np.linalg.norm(pts[9])

    if scale < 1e-8:
        scale = 1.0

    pts = pts / scale

    f = {}

    for i in range(21):
        f[f"x_{i}"] = pts[i,0]
        f[f"y_{i}"] = pts[i,1]
        f[f"z_{i}"] = pts[i,2]

    tips = {
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20
    }

    mcps = {
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17
    }

    # Finger extension ratios
    for name, tip in tips.items():

        mcp = mcps[name]

        f[f"{name}_ratio"] = (
            np.linalg.norm(pts[tip]) /
            max(np.linalg.norm(pts[mcp]), 1e-8)
        )

    # Fingertip pair features
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
        [f[name] for name in FEATURES],
        dtype=float
    )

print("="*72)
print("V16 STEP 10 - TRUE BALANCED LIVE VALIDATION")
print("="*72)

cal = pd.read_csv(CAL)
test = pd.read_csv(TEST)

print(f"Calibration samples : {len(cal)}")
print(f"Validation samples  : {len(test)}")

print("\nValidation distribution:")
print(test["label"].value_counts())

# Generate identical feature representation for BOTH datasets
X_train = np.vstack([
    make_features(row)
    for _, row in cal.iterrows()
])

X_test = np.vstack([
    make_features(row)
    for _, row in test.iterrows()
])

y_train = cal["label"].values
y_test = test["label"].values

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

print(f"\nFeatures used: {X_train.shape[1]}")

# Standardization learned ONLY from calibration data
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Model trained ONLY on calibration data
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=1.0
)

model.fit(X_train_s, y_train)

pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)

accuracy = accuracy_score(y_test, pred)

print("\n" + "="*72)
print("TRUE UNSEEN LIVE VALIDATION RESULTS")
print("="*72)

print(f"Overall accuracy: {accuracy*100:.2f}%")
print()

print(classification_report(
    y_test,
    pred,
    labels=["one","two","three"],
    zero_division=0
))

print("="*72)
print("CONFUSION MATRIX")
print("="*72)

cm = confusion_matrix(
    y_test,
    pred,
    labels=["one","two","three"]
)

print("              ONE  TWO  THREE")

for name,row in zip(
    ["ONE","TWO","THREE"],
    cm
):
    print(
        f"{name:8s} "
        f"{row[0]:4d} "
        f"{row[1]:4d} "
        f"{row[2]:6d}"
    )

print("\n" + "="*72)
print("PER-GESTURE ACCURACY")
print("="*72)

for gesture in ["one","two","three"]:

    mask = y_test == gesture

    correct = (
        pred[mask] == gesture
    ).sum()

    total = mask.sum()

    print(
        f"{gesture.upper():8s}: "
        f"{correct/total*100:6.2f}% "
        f"({correct}/{total})"
    )

print("\n" + "="*72)
print("PREDICTION DISTRIBUTION")
print("="*72)

print(pd.Series(pred).value_counts())

print("\n" + "="*72)
print("TWO-SPECIFIC RESULT")
print("="*72)

two_mask = y_test == "two"

two_correct = (
    pred[two_mask] == "two"
).sum()

print(f"Actual TWO samples : {two_mask.sum()}")
print(f"Correct TWO        : {two_correct}")
print(
    f"TWO recall         : "
    f"{two_correct/two_mask.sum()*100:.2f}%"
)

print("\nTWO predictions:")

vals, counts = np.unique(
    pred[two_mask],
    return_counts=True
)

for v,c in zip(vals,counts):
    print(f"  {v.upper():8s}: {c}")

# Confidence analysis
sorted_probs = np.sort(proba, axis=1)

margin = (
    sorted_probs[:,-1] -
    sorted_probs[:,-2]
)

print("\n" + "="*72)
print("CONFIDENCE REJECTION")
print("="*72)

for threshold in [0.10,0.20,0.30,0.40,0.50,0.60]:

    accepted = margin >= threshold

    coverage = accepted.mean()

    if accepted.sum():

        acc = (
            pred[accepted] ==
            y_test[accepted]
        ).mean()

    else:

        acc = 0

    print(
        f"Margin >= {threshold:.2f} | "
        f"Coverage={coverage*100:6.2f}% | "
        f"Accuracy={acc*100:6.2f}% | "
        f"Accepted={accepted.sum():3d}"
    )

print("\n" + "="*72)
print("INTERPRETATION")
print("="*72)

if accuracy >= 0.80:
    print("STRONG: recognition is becoming reliable.")

elif accuracy >= 0.60:
    print("PROMISING: useful recognition, but refinement is needed.")

elif accuracy >= 0.40:
    print("PARTIAL: recognition works, but significant errors remain.")

else:
    print("WEAK: the current feature/decision representation needs improvement.")

print()
print("IMPORTANT:")
print("This 300-sample dataset was NOT used for training.")
print("It is a genuine live validation set.")

print("\n" + "="*72)
print("V16 STEP 10 BALANCED VALIDATION COMPLETE")
print("="*72)

input("Press Enter to return to PowerShell...")
