from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset" / "gesture_dataset_v8_features.csv"
MODEL_DIR = BASE_DIR / "trained_model"
REPORT = BASE_DIR / "dataset" / "v8_training_results.txt"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATASET)

feature_cols = [
    c for c in df.columns
    if c.startswith("f_")
]

print("=" * 75)
print("V8 UNSEEN-SESSION MODEL EVALUATION")
print("=" * 75)

print(f"\nRows: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Sessions: {df['session_id'].nunique()}")

print("\nGesture distribution:")
print(df["label"].value_counts().sort_index())

# ------------------------------------------------------------
# Identify sessions
# ------------------------------------------------------------

session_info = (
    df.groupby("session_id")["label"]
      .first()
      .reset_index()
)

print("\nSESSION LIST")

for _, row in session_info.iterrows():

    print(
        f"{row['label']:12} -> {row['session_id']}"
    )

# ------------------------------------------------------------
# Select first and second session per gesture
# ------------------------------------------------------------

train_sessions = {}
test_sessions = {}

for gesture in sorted(df["label"].unique()):

    sessions = (
        df.loc[
            df["label"] == gesture,
            "session_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    if len(sessions) < 2:

        raise ValueError(
            f"Gesture '{gesture}' does not have 2 sessions."
        )

    train_sessions[gesture] = sessions[0]
    test_sessions[gesture] = sessions[1]

print("\nTRAINING SESSIONS")

for gesture, session in train_sessions.items():

    print(
        f"{gesture:12} -> {session}"
    )

print("\nUNSEEN TEST SESSIONS")

for gesture, session in test_sessions.items():

    print(
        f"{gesture:12} -> {session}"
    )

# ------------------------------------------------------------
# Split
# ------------------------------------------------------------

train_mask = df.apply(
    lambda row:
        train_sessions[row["label"]] == row["session_id"],
    axis=1
)

test_mask = df.apply(
    lambda row:
        test_sessions[row["label"]] == row["session_id"],
    axis=1
)

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

X_train = train_df[feature_cols].to_numpy()
X_test = test_df[feature_cols].to_numpy()

y_train = train_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

print("\nDATA SPLIT")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

models = {

    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

# ------------------------------------------------------------
# Train/evaluate
# ------------------------------------------------------------

for name, model in models.items():

    print("\n" + "-" * 75)
    print(f"TRAINING: {name}")
    print("-" * 75)

    if name == "SVM":

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = (
        model,
        accuracy,
        predictions
    )

    print(
        f"{name} accuracy: "
        f"{accuracy * 100:.2f}%"
    )

# ------------------------------------------------------------
# Best model
# ------------------------------------------------------------

best_name = max(
    results,
    key=lambda name: results[name][1]
)

best_model, best_accuracy, best_predictions = results[
    best_name
]

print("\n" + "=" * 75)
print("V8 RESULTS")
print("=" * 75)

for name in results:

    print(
        f"{name:15}: "
        f"{results[name][1] * 100:.2f}%"
    )

print(
    f"\nBEST MODEL: {best_name}"
)

print(
    f"BEST ACCURACY: "
    f"{best_accuracy * 100:.2f}%"
)

# ------------------------------------------------------------
# Detailed report
# ------------------------------------------------------------

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_test,
    best_predictions,
    digits=4,
    zero_division=0
)

print(report)

print("CONFUSION MATRIX")

labels = sorted(
    df["label"].unique()
)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=labels
)

print(
    pd.DataFrame(
        cm,
        index=[
            f"Actual_{x}"
            for x in labels
        ],
        columns=[
            f"Pred_{x}"
            for x in labels
        ]
    )
)

# ------------------------------------------------------------
# Per-gesture accuracy
# ------------------------------------------------------------

print("\nPER-GESTURE ACCURACY")

for gesture in labels:

    mask = y_test == gesture

    if mask.sum() == 0:
        continue

    gesture_accuracy = np.mean(
        best_predictions[mask] == gesture
    )

    print(
        f"{gesture:12}: "
        f"{gesture_accuracy * 100:.2f}% "
        f"({np.sum(best_predictions[mask] == gesture)}/{mask.sum()})"
    )

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

import joblib

if best_name == "SVM":

    joblib.dump(
        best_model,
        MODEL_DIR / "gesture_svm_v8.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler_v8.pkl"
    )

else:

    filename = (
        "gesture_rf_v8.pkl"
        if best_name == "Random Forest"
        else "gesture_extra_trees_v8.pkl"
    )

    joblib.dump(
        best_model,
        MODEL_DIR / filename
    )

print("\nModel saved.")

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V8 UNSEEN-SESSION MODEL EVALUATION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Features: {len(feature_cols)}\n"
    )

    f.write(
        f"Training samples: {len(X_train)}\n"
    )

    f.write(
        f"Testing samples: {len(X_test)}\n\n"
    )

    for name in results:

        f.write(
            f"{name}: "
            f"{results[name][1] * 100:.2f}%\n"
        )

    f.write(
        f"\nBest model: {best_name}\n"
    )

    f.write(
        f"Best accuracy: "
        f"{best_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        report
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        ).to_string()
    )

print(
    f"\nReport saved to:\n{REPORT}"
)

print("\n" + "=" * 75)
print("V8 TRAINING COMPLETE")
print("=" * 75)
