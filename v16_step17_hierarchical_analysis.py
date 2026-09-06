import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
CAL = os.path.join(BASE, "dataset", "v16_calibration", "v16_live_calibration_one_two_three.csv")
VAL = os.path.join(BASE, "dataset", "v16_calibration", "v16_balanced_live_validation.csv")
OUT = os.path.join(BASE, "dataset", "v16_step17_hierarchical_results.txt")

def load_raw(path):
    df = pd.read_csv(path)
    feature_cols = [f"{a}_{i}" for i in range(21) for a in ["x","y","z"]]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values
    return X, y

def direction_features(X):
    X = X.reshape(-1, 21, 3)

    # MediaPipe finger chains
    fingers = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9,10,11,12],
        "ring":   [13,14,15,16],
        "pinky":  [17,18,19,20],
    }

    all_features = {}
    for name, idx in fingers.items():
        pts = X[:, idx, :]
        bones = []
        for j in range(3):
            v = pts[:, j+1] - pts[:, j]
            n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            bones.append(v / n)

        # 9 directional features per finger
        f = np.concatenate(bones, axis=1)
        all_features[name] = f

    groups = {}

    order = ["thumb","index","middle","ring","pinky"]

    for name in order:
        groups[name] = all_features[name]

    groups["middle+ring"] = np.concatenate(
        [all_features["middle"], all_features["ring"]], axis=1
    )

    groups["middle+ring+pinky"] = np.concatenate(
        [all_features["middle"], all_features["ring"], all_features["pinky"]], axis=1
    )

    groups["index+middle+ring+pinky"] = np.concatenate(
        [all_features["index"], all_features["middle"],
         all_features["ring"], all_features["pinky"]], axis=1
    )

    groups["all_fingers"] = np.concatenate(
        [all_features[n] for n in order], axis=1
    )

    return groups

Xcal_raw, ycal = load_raw(CAL)
Xval_raw, yval = load_raw(VAL)

cal_groups = direction_features(Xcal_raw)
val_groups = direction_features(Xval_raw)

groups_to_test = [
    "index",
    "middle",
    "ring",
    "pinky",
    "middle+ring",
    "middle+ring+pinky",
    "index+middle+ring+pinky",
    "all_fingers"
]

ks = [1,3,5,7,9,11,15]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------
# ONE vs NON-ONE
# ---------------------------------------------------------

ycal_one = np.asarray(np.where(ycal == "one", "one", "non_one"))

one_results = []

for group in groups_to_test:
    Xg = cal_groups[group]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xg)

    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        scores = cross_val_score(
            model, Xs, ycal_one,
            cv=cv,
            scoring="balanced_accuracy"
        )

        one_results.append({
            "group": group,
            "k": k,
            "cv_balanced_accuracy": scores.mean()
        })

one_df = pd.DataFrame(one_results).sort_values(
    "cv_balanced_accuracy", ascending=False
)

best_one = one_df.iloc[0]

print("\n" + "="*70)
print("STEP 17 — ONE vs NON-ONE")
print("="*70)
print(one_df.head(10).to_string(index=False))

best_one_group = best_one["group"]
best_one_k = int(best_one["k"])

# ---------------------------------------------------------
# TWO vs THREE
# ---------------------------------------------------------

mask23 = np.isin(ycal, ["two","three"])
ycal_23 = np.asarray(ycal[mask23])

two_three_results = []

for group in groups_to_test:
    Xg = cal_groups[group][mask23]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xg)

    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        scores = cross_val_score(
            model, Xs, ycal_23,
            cv=cv,
            scoring="balanced_accuracy"
        )

        two_three_results.append({
            "group": group,
            "k": k,
            "cv_balanced_accuracy": scores.mean()
        })

two_three_df = pd.DataFrame(two_three_results).sort_values(
    "cv_balanced_accuracy", ascending=False
)

best_23 = two_three_df.iloc[0]

print("\n" + "="*70)
print("STEP 17 — TWO vs THREE")
print("="*70)
print(two_three_df.head(10).to_string(index=False))

best_23_group = best_23["group"]
best_23_k = int(best_23["k"])

# ---------------------------------------------------------
# TRAIN FINAL MODELS ON ALL CALIBRATION DATA
# ---------------------------------------------------------

one_scaler = StandardScaler()
X_one_train = one_scaler.fit_transform(cal_groups[best_one_group])

one_model = KNeighborsClassifier(
    n_neighbors=best_one_k,
    weights="distance"
)
one_model.fit(X_one_train, ycal_one)

mask23 = np.isin(ycal, ["two","three"])

two_scaler = StandardScaler()
X_23_train = two_scaler.fit_transform(
    cal_groups[best_23_group][mask23]
)

two_model = KNeighborsClassifier(
    n_neighbors=best_23_k,
    weights="distance"
)
two_model.fit(X_23_train, ycal_23)

# ---------------------------------------------------------
# HIERARCHICAL VALIDATION
# ---------------------------------------------------------

X_one_val = one_scaler.transform(val_groups[best_one_group])
one_pred = one_model.predict(X_one_val)

X_23_val = two_scaler.transform(
    val_groups[best_23_group]
)

two_pred = two_model.predict(X_23_val)

final_pred = []

for i in range(len(yval)):
    if one_pred[i] == "one":
        final_pred.append("one")
    else:
        final_pred.append(two_pred[i])

final_pred = np.array(final_pred)

acc = accuracy_score(yval, final_pred)
bal = balanced_accuracy_score(yval, final_pred)
cm = confusion_matrix(
    yval,
    final_pred,
    labels=["one","two","three"]
)

print("\n" + "="*70)
print("HIERARCHICAL VALIDATION")
print("="*70)

print(f"ONE detector: {best_one_group}, k={best_one_k}")
print(f"TWO/THREE classifier: {best_23_group}, k={best_23_k}")
print(f"\nAccuracy: {acc*100:.2f}%")
print(f"Balanced Accuracy: {bal*100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        yval,
        final_pred,
        labels=["one","two","three"],
        zero_division=0
    )
)

for label in ["one","two","three"]:
    total = np.sum(yval == label)
    correct = np.sum((yval == label) & (final_pred == label))
    print(f"{label.upper():5s}: {correct}/{total} = {correct/total*100:.2f}%")

one_total = np.sum(yval == "one")
one_correct = np.sum((yval == "one") & (final_pred == "one"))

non_one_total = np.sum(yval != "one")
false_one = np.sum((yval != "one") & (final_pred == "one"))

print(f"\nONE recall: {one_correct/one_total*100:.2f}%")
print(f"ONE false-positive rate: {false_one/non_one_total*100:.2f}%")

# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

with open(OUT, "w", encoding="utf-8") as f:
    f.write("STEP 17 — HIERARCHICAL GESTURE RECOGNITION\n")
    f.write("="*70 + "\n\n")

    f.write("Purpose:\n")
    f.write("Specialized ONE detector + TWO/THREE classifier.\n")
    f.write("Model selection uses calibration data only.\n")
    f.write("Final validation uses unseen balanced validation data.\n\n")

    f.write("ONE vs NON-ONE best configuration:\n")
    f.write(f"Feature group: {best_one_group}\n")
    f.write(f"KNN k: {best_one_k}\n")
    f.write(f"5-fold CV balanced accuracy: {best_one['cv_balanced_accuracy']*100:.2f}%\n\n")

    f.write("TWO vs THREE best configuration:\n")
    f.write(f"Feature group: {best_23_group}\n")
    f.write(f"KNN k: {best_23_k}\n")
    f.write(f"5-fold CV balanced accuracy: {best_23['cv_balanced_accuracy']*100:.2f}%\n\n")

    f.write("UNSEEN VALIDATION RESULTS\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Balanced Accuracy: {bal*100:.2f}%\n\n")

    f.write("Confusion Matrix [one,two,three]:\n")
    f.write(str(cm) + "\n\n")

    f.write(
        classification_report(
            yval,
            final_pred,
            labels=["one","two","three"],
            zero_division=0
        )
    )

    f.write("\n")
    f.write(f"ONE recall: {one_correct/one_total*100:.2f}%\n")
    f.write(f"ONE false-positive rate: {false_one/non_one_total*100:.2f}%\n")

print(f"\nReport saved: {OUT}")
