import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
DATASET = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")

print("=" * 75)
print("V9 DATASET DIAGNOSTIC")
print("=" * 75)

df = pd.read_csv(DATASET)

feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Features: {len(feature_cols)}")
print(f"Gestures: {df['label'].nunique()}")
print(f"Sessions: {df['session_id'].nunique()}")

print("\nGESTURE DISTRIBUTION")
print(df["label"].value_counts().sort_index())

print("\nSESSION DISTRIBUTION")
print(df.groupby(["label", "session_id"]).size())

X = df[feature_cols].values

print("\nDATA QUALITY")
print(f"Missing values: {df[feature_cols].isna().sum().sum()}")
print(f"Non-finite values: {np.isfinite(X).all()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\nRAW FEATURE RANGE")
print(f"Minimum: {X.min():.6f}")
print(f"Maximum: {X.max():.6f}")
print(f"Mean: {X.mean():.6f}")
print(f"Std: {X.std():.6f}")

print("\nSESSION CORRELATION ANALYSIS")

session_means = df.groupby("session_id")[feature_cols].mean()

correlations = []

for feature in feature_cols:
    values = session_means[feature].values

    if np.std(values) > 0:
        corr = np.std(values)
    else:
        corr = 0

    correlations.append((feature, corr))

correlations.sort(key=lambda x: x[1], reverse=True)

print("\nMost session-variable features:")
for feature, value in correlations[:15]:
    print(f"{feature:8s} -> {value:.6f}")

print("\nGESTURE CENTROID DISTANCES")

gesture_means = df.groupby("label")[feature_cols].mean()

labels = sorted(gesture_means.index)

pairs = []

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):

        a = gesture_means.loc[labels[i]].values
        b = gesture_means.loc[labels[j]].values

        distance = np.linalg.norm(a - b)

        pairs.append(
            (labels[i], labels[j], distance)
        )

pairs.sort(key=lambda x: x[2])

for a, b, distance in pairs:
    print(f"{a:10s} <-> {b:10s} : {distance:.6f}")

print("\n" + "=" * 75)
print("V9 DIAGNOSTIC COMPLETE")
print("=" * 75)
