import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
TEST = BASE / "dataset" / "v16_calibration" / "v16_step9n_fresh_validation.csv"

print("="*72)
print("V16 DATA AUDIT - ONE / TWO / THREE")
print("="*72)

cal = pd.read_csv(CAL)
test = pd.read_csv(TEST)

print("\nCALIBRATION")
print("-"*72)
print("Shape:", cal.shape)
print("Columns:", list(cal.columns))
print("\nLabels:")
print(cal["label"].value_counts())

print("\nFRESH TEST")
print("-"*72)
print("Shape:", test.shape)
print("Columns:", list(test.columns))

if "label" in test.columns:
    print("\nLabels:")
    print(test["label"].value_counts())

print("\n" + "="*72)
print("FEATURE FORMAT CHECK")
print("="*72)

landmark_cols = [
    f"{axis}_{i}"
    for i in range(21)
    for axis in ["x","y","z"]
]

cal_landmarks = [c for c in landmark_cols if c in cal.columns]
test_landmarks = [c for c in landmark_cols if c in test.columns]

print("Calibration landmark columns:", len(cal_landmarks), "/ 63")
print("Fresh-test landmark columns :", len(test_landmarks), "/ 63")

print("\nCalibration missing:")
print([c for c in landmark_cols if c not in cal.columns])

print("\nFresh-test missing:")
print([c for c in landmark_cols if c not in test.columns])

print("\n" + "="*72)
print("NUMERIC VALIDITY")
print("="*72)

for name,df,cols in [
    ("CALIBRATION",cal,cal_landmarks),
    ("FRESH TEST",test,test_landmarks)
]:

    if cols:

        X = df[cols].apply(pd.to_numeric, errors="coerce").values

        print(f"\n{name}")
        print("NaN values :", np.isnan(X).sum())
        print("Inf values :", np.isinf(X).sum())
        print("Min        :", np.nanmin(X))
        print("Max        :", np.nanmax(X))
        print("Mean       :", np.nanmean(X))
        print("Std        :", np.nanstd(X))

print("\n" + "="*72)
print("PER-GESTURE RAW LANDMARK STATISTICS")
print("="*72)

for gesture in ["one","two","three"]:

    print(f"\n--- {gesture.upper()} ---")

    subset = cal[cal["label"] == gesture]

    if len(subset) == 0:
        continue

    for c in [
        "x_8","y_8",
        "x_12","y_12",
        "x_16","y_16",
        "x_20","y_20"
    ]:

        if c in subset.columns:

            values = pd.to_numeric(
                subset[c],
                errors="coerce"
            )

            print(
                f"{c:5s}: "
                f"mean={values.mean(): .4f} "
                f"std={values.std(): .4f} "
                f"min={values.min(): .4f} "
                f"max={values.max(): .4f}"
            )

print("\n" + "="*72)
print("FRESH TEST GESTURE STATISTICS")
print("="*72)

if "label" in test.columns:

    for gesture in ["one","two","three"]:

        subset = test[test["label"] == gesture]

        print(
            f"\n{gesture.upper()}: "
            f"{len(subset)} samples"
        )

        for c in [
            "x_8","y_8",
            "x_12","y_12",
            "x_16","y_16",
            "x_20","y_20"
        ]:

            if c in subset.columns and len(subset):

                values = pd.to_numeric(
                    subset[c],
                    errors="coerce"
                )

                print(
                    f"{c:5s}: "
                    f"mean={values.mean(): .4f} "
                    f"std={values.std(): .4f}"
                )

print("\n" + "="*72)
print("IMPORTANT")
print("="*72)

if len(cal_landmarks) == 63:
    print("Calibration: RAW LANDMARKS OK")
else:
    print("Calibration: RAW LANDMARKS INCOMPLETE")

if len(test_landmarks) == 63:
    print("Fresh test: RAW LANDMARKS OK")
else:
    print("Fresh test: engineered/partial features detected")

print("\nAudit complete.")
input("\nPress Enter to return to PowerShell...")
