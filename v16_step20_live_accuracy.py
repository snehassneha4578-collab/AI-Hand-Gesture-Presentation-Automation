import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
CAL = os.path.join(BASE, "dataset", "v16_calibration", "v16_live_calibration_one_two_three.csv")
TASK = os.path.join(BASE, "models", "hand_landmarker.task")
REPORT = os.path.join(BASE, "dataset", "v16_step20_live_accuracy_results.txt")

# ---------------------------------------------------------
# LOAD CALIBRATION DATA
# ---------------------------------------------------------

print("STEP 20 STARTING...")
print("Loading calibration data...")

df = pd.read_csv(CAL)

cols = [f"{a}_{i}" for i in range(21) for a in ["x","y","z"]]

X = df[cols].values.astype(np.float64)
y = np.asarray(df["label"].values)

X = X.reshape(-1,21,3)

fingers = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20]
]

def direction_features(data):

    output = []

    for sample in data:

        features = []

        for finger in fingers:

            pts = sample[finger]

            for j in range(3):

                v = pts[j+1] - pts[j]

                norm = np.sqrt(
                    v[0]**2 +
                    v[1]**2 +
                    v[2]**2
                ) + 1e-8

                features.extend([
                    v[0]/norm,
                    v[1]/norm,
                    v[2]/norm
                ])

        output.append(features)

    return np.asarray(output,dtype=np.float64)

features = direction_features(X)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

model.fit(features_scaled,y)

print("Model ready.")

# ---------------------------------------------------------
# MEDIAPIPE
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    landmarker.close()
    raise SystemExit

# ---------------------------------------------------------
# TEST INSTRUCTIONS
# ---------------------------------------------------------

gestures = ["one","two","three"]

print("")
print("="*70)
print("CONTROLLED LIVE ACCURACY TEST")
print("="*70)
print("")
print("You will test 10 samples of each gesture.")
print("")
print("For every trial:")
print("")
print("  1. Look at the screen.")
print("  2. Make the requested gesture.")
print("  3. HOLD it steadily.")
print("  4. Press SPACE when ready.")
print("  5. The system records ONE prediction.")
print("  6. Release your hand before the next trial.")
print("")
print("Order:")
print("ONE × 10")
print("TWO × 10")
print("THREE × 10")
print("")
print("Press Q anytime to quit.")
print("="*70)

results = []

for true_label in gestures:

    for trial in range(1,11):

        print("")
        print(
            f"READY: {true_label.upper()} "
            f"trial {trial}/10"
        )

        captured = False
        prediction = "NO_HAND"
        confidence = 0.0

        while not captured:

            ret, frame = cap.read()

            if not ret:
                print("Camera frame error.")
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(image)

            live_prediction = "NO_HAND"
            live_confidence = 0.0

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                coords = np.array(
                    [[p.x,p.y,p.z] for p in landmarks],
                    dtype=np.float64
                )

                live = direction_features(
                    coords.reshape(1,21,3)
                )

                live_scaled = scaler.transform(live)

                probs = model.predict_proba(
                    live_scaled
                )[0]

                order = np.argsort(probs)[::-1]

                best = order[0]
                second = order[1]

                live_prediction = model.classes_[best]

                p1 = probs[best]
                p2 = probs[second]

                margin = p1-p2

                live_confidence = (
                    0.6*p1 +
                    0.4*margin
                )

            cv2.putText(
                frame,
                f"TRUE: {true_label.upper()}",
                (20,45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,255),
                2
            )

            cv2.putText(
                frame,
                f"PREDICTION: {live_prediction.upper()}",
                (20,85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"CONFIDENCE: {live_confidence:.2f}",
                (20,125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                "SPACE = RECORD | Q = QUIT",
                (20,165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

            cv2.imshow(
                "STEP 20 - Controlled Accuracy Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                cap.release()
                cv2.destroyAllWindows()
                landmarker.close()
                print("Test cancelled.")
                raise SystemExit

            if key == 32:

                prediction = live_prediction
                confidence = live_confidence
                captured = True

        correct = prediction == true_label

        results.append({
            "true": true_label,
            "predicted": prediction,
            "confidence": confidence,
            "correct": correct
        })

        print(
            f"Recorded: TRUE={true_label.upper()} "
            f"| PRED={prediction.upper()} "
            f"| CONF={confidence:.2f} "
            f"| {'CORRECT' if correct else 'WRONG'}"
        )

        time.sleep(0.7)

cap.release()
cv2.destroyAllWindows()
landmarker.close()

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

res = pd.DataFrame(results)

print("")
print("="*70)
print("STEP 20 RESULTS")
print("="*70)

overall = res["correct"].mean()*100

print(f"Overall accuracy: {overall:.2f}%")

print("")

for label in gestures:

    subset = res[res["true"] == label]

    acc = subset["correct"].mean()*100

    print(
        f"{label.upper():5s}: "
        f"{int(subset['correct'].sum())}/10 "
        f"= {acc:.2f}%"
    )

print("")
print("Confusion Matrix:")

cm = pd.crosstab(
    res["true"],
    res["predicted"],
    rownames=["True"],
    colnames=["Predicted"]
)

print(cm)

print("")
print("All trials:")
print(res.to_string(index=False))

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

with open(REPORT,"w",encoding="utf-8") as f:

    f.write("STEP 20 — CONTROLLED REAL-TIME ACCURACY TEST\n")
    f.write("="*70+"\n\n")

    f.write(
        f"Overall accuracy: {overall:.2f}%\n\n"
    )

    for label in gestures:

        subset = res[res["true"] == label]

        f.write(
            f"{label.upper()}: "
            f"{int(subset['correct'].sum())}/10 "
            f"= {subset['correct'].mean()*100:.2f}%\n"
        )

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nAll trials:\n")
    f.write(res.to_string(index=False))

print("")
print(f"Report saved: {REPORT}")
print("="*70)
