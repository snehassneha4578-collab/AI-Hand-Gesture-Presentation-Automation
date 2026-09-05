import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

BASE = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
MODEL = BASE + r"\trained_model\v16_live_prototypes.pkl"
TASK_MODEL = BASE + r"\models\hand_landmarker.task"

with open(MODEL, "rb") as f:
    prototypes = pickle.load(f)

GESTURES = ["one", "two", "three"]

def extract_features(hand):
    pts = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)

    pts -= pts[0]

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0

    pts /= scale

    features = list(pts.flatten())

    fingertips = [4, 8, 12, 16, 20]

    for idx in fingertips:
        features.append(np.linalg.norm(pts[idx]))

    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(
                np.linalg.norm(
                    pts[fingertips[i]] - pts[fingertips[j]]
                )
            )

    for idx in fingertips:
        features.extend(pts[idx].tolist())

    for idx in [8, 12, 16, 20]:
        features.append(np.linalg.norm(pts[4] - pts[idx]))

    return np.array(features, dtype=np.float32)

def distance(features, prototype):
    z = (features - prototype["mean"]) / prototype["std"]
    return np.linalg.norm(z)

print("=" * 72)
print("V16 STEP 9J - FRESH LIVE ACCURACY EVALUATION")
print("=" * 72)
print()
print("You will test each gesture separately.")
print()
print("1. Press O = start ONE test")
print("2. Hold ONE in front of camera for about 10 seconds")
print("3. Press T = start TWO test")
print("4. Hold TWO for about 10 seconds")
print("5. Press H = start THREE test")
print("6. Hold THREE for about 10 seconds")
print("7. Press Q = finish")
print()
print("Only use the requested gesture during each test.")
print("=" * 72)

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_MODEL),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    input("Press Enter...")
    raise SystemExit

actual = None
results = {g: [] for g in GESTURES}
start_time = None

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        prediction = "NO HAND"
        margin = 0.0

        if result.hand_landmarks:

            features = extract_features(result.hand_landmarks[0])

            distances = {
                g: distance(features, prototypes[g])
                for g in GESTURES
            }

            ranked = sorted(
                distances.items(),
                key=lambda x: x[1]
            )

            prediction = ranked[0][0]
            margin = ranked[1][1] - ranked[0][1]

            if actual is not None:
                results[actual].append(prediction)

        if actual is not None:
            elapsed = time.time() - start_time
            remaining = max(0, 10 - elapsed)

            cv2.putText(
                frame,
                f"TESTING: {actual.upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Prediction: {prediction.upper()}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Time remaining: {remaining:.1f}s",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if elapsed >= 10:
                actual = None
                start_time = None

        else:
            cv2.putText(
                frame,
                "Press O=ONE  T=TWO  H=THREE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.imshow("V16 Step 9J - Accuracy Evaluation", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("o"):
            actual = "one"
            results["one"] = []
            start_time = time.time()
            print("ONE test started...")

        elif key == ord("t"):
            actual = "two"
            results["two"] = []
            start_time = time.time()
            print("TWO test started...")

        elif key == ord("h"):
            actual = "three"
            results["three"] = []
            start_time = time.time()
            print("THREE test started...")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 72)
print("V16 STEP 9J RESULTS")
print("=" * 72)

total_correct = 0
total_samples = 0

for g in GESTURES:

    preds = results[g]

    if not preds:
        print(f"{g.upper():5s}: No samples")
        continue

    correct = sum(p == g for p in preds)
    accuracy = correct / len(preds) * 100

    total_correct += correct
    total_samples += len(preds)

    print(
        f"{g.upper():5s}: "
        f"{accuracy:6.2f}% "
        f"({correct}/{len(preds)})"
    )

    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1

    print("       Predictions:", counts)

print("-" * 72)

if total_samples:
    overall = total_correct / total_samples * 100
    print(f"OVERALL ACCURACY: {overall:.2f}%")
    print(f"TOTAL SAMPLES:    {total_samples}")
else:
    print("No evaluation samples collected.")

print("=" * 72)

input("Press Enter to return to PowerShell...")
