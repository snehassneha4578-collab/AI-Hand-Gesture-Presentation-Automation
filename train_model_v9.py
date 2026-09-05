import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
REPORT_PATH = os.path.join(BASE_DIR, "dataset", "v9_training_results.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 75)
print("V9 UNSEEN-SESSION MODEL EVALUATION")
print("=" * 75)

df = pd.read_csv(DATASET)

feature_cols = [
    c for c in df.columns
    if c.startswith(("x_", "y_", "z_"))
]

labels = sorted(df["label"].unique())

print(f"\nRows: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Sessions: {df['session_id'].nunique()}")

print("\nGESTURE DISTRIBUTION")
print(df["label"].value_counts().sort_index())

# ---------------------------------------------------------
# SESSION SPLIT
# ---------------------------------------------------------
# Session 1 + Session 2 = training
# Session 3 = completely unseen testing
# ---------------------------------------------------------

train_parts = []
test_parts = []

print("\nSESSION SPLIT")

for gesture in labels:

    sessions = list(
        df[df["label"] == gesture]["session_id"].unique()
    )

    sessions = sorted(sessions)

    if len(sessions) < 3:
        raise ValueError(
            f"{gesture} has fewer than 3 sessions."
        )

    train_sessions = sessions[:2]
    test_session = sessions[2]

    print(f"\n{gesture}")
    print(f"  Training: {train_sessions}")
    print(f"  Testing : {test_session}")

    train_parts.append(
        df[
            (df["label"] == gesture) &
            (df["session_id"].isin(train_sessions))
        ]
    )

    test_parts.append(
        df[
            (df["label"] == gesture) &
            (df["session_id"] == test_session)
        ]
    )

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

X_train = train_df[feature_cols].values
y_train = train_df["label"].values

X_test = test_df[feature_cols].values
y_test = test_df["label"].values

print("\nDATA SPLIT")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("TRAINING: SVM")
print("-" * 75)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm.fit(X_train_scaled, y_train)

svm_pred = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"SVM accuracy: {svm_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("TRAINING: Random Forest")
print("-" * 75)

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Random Forest accuracy: {rf_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# EXTRA TREES
# ---------------------------------------------------------

print("\n" + "-" * 75)
print("TRAINING: Extra Trees")
print("-" * 75)

extra = ExtraTreesClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

extra.fit(X_train, y_train)

extra_pred = extra.predict(X_test)
extra_accuracy = accuracy_score(y_test, extra_pred)

print(f"Extra Trees accuracy: {extra_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# SELECT BEST
# ---------------------------------------------------------

results = {
    "SVM": (svm, svm_accuracy),
    "Random Forest": (rf, rf_accuracy),
    "Extra Trees": (extra, extra_accuracy)
}

best_name = max(
    results,
    key=lambda name: results[name][1]
)

best_model = results[best_name][0]
best_accuracy = results[best_name][1]

if best_name == "SVM":

    joblib.dump(
        svm,
        os.path.join(MODEL_DIR, "gesture_svm_v9.pkl")
    )

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "scaler_v9.pkl")
    )

elif best_name == "Random Forest":

    joblib.dump(
        rf,
        os.path.join(MODEL_DIR, "gesture_rf_v9.pkl")
    )

else:

    joblib.dump(
        extra,
        os.path.join(MODEL_DIR, "gesture_extra_trees_v9.pkl")
    )

# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------

if best_name == "SVM":
    best_pred = svm_pred
elif best_name == "Random Forest":
    best_pred = rf_pred
else:
    best_pred = extra_pred

report = classification_report(
    y_test,
    best_pred,
    labels=labels,
    target_names=labels,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    best_pred,
    labels=labels
)

print("\n" + "=" * 75)
print("V9 RESULTS")
print("=" * 75)

print(f"SVM            : {svm_accuracy * 100:.2f}%")
print(f"Random Forest  : {rf_accuracy * 100:.2f}%")
print(f"Extra Trees    : {extra_accuracy * 100:.2f}%")

print(f"\nBEST MODEL: {best_name}")
print(f"BEST ACCURACY: {best_accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(report)

print("CONFUSION MATRIX")

print(
    pd.DataFrame(
        cm,
        index=[f"Actual_{x}" for x in labels],
        columns=[f"Pred_{x}" for x in labels]
    ).to_string()
)

print("\nPER-GESTURE ACCURACY")

for i, label in enumerate(labels):

    total = cm[i].sum()

    correct = cm[i, i]

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{label:10s}: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("V9 UNSEEN-SESSION MODEL EVALUATION\n")
    f.write("=" * 75 + "\n\n")

    f.write(f"Rows: {len(df)}\n")
    f.write(f"Features: {len(feature_cols)}\n")
    f.write(f"Sessions: {df['session_id'].nunique()}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n\n")

    f.write(f"SVM: {svm_accuracy * 100:.2f}%\n")
    f.write(
        f"Random Forest: "
        f"{rf_accuracy * 100:.2f}%\n"
    )
    f.write(
        f"Extra Trees: "
        f"{extra_accuracy * 100:.2f}%\n"
    )

    f.write(f"\nBest model: {best_name}\n")
    f.write(
        f"Best accuracy: "
        f"{best_accuracy * 100:.2f}%\n\n"
    )

    f.write(report)

    f.write("\nConfusion Matrix:\n")

    f.write(
        pd.DataFrame(
            cm,
            index=[f"Actual_{x}" for x in labels],
            columns=[f"Pred_{x}" for x in labels]
        ).to_string()
    )

print(f"\nModel saved in:\n{MODEL_DIR}")

print(f"\nReport saved to:\n{REPORT_PATH}")

print("\n" + "=" * 75)
print("V9 TRAINING COMPLETE")
print("=" * 75)
