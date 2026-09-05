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

def normalize(sample):
    pts = sample.reshape(21, 3).copy()
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
        features.append(
            np.linalg.norm(pts[4] - pts[idx])
        )

    return np.array(features, dtype=np.float32)

def score(sample, prototype):
    z = (sample - prototype["mean"]) / prototype["std"]
    return np.linalg.norm(z)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_MODEL),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    input("Press Enter to exit...")
    raise SystemExit

print("=" * 70)
print("V16 STEP 9I - LIVE VALIDATION")
print("=" * 70)
print("Show ONE, TWO, and THREE gestures in front of the camera.")
print("Watch the prediction and margin.")
print("Press Q to quit.")
print("=" * 70)

frame_count = 0
hand_frames = 0
start = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ok, frame = cap.read()

        if not ok:
            continue

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        prediction = "NO HAND"
        margin = 0.0
        best_distance = 0.0

        if result.hand_landmarks:

            hand_frames += 1
            hand = result.hand_landmarks[0]

            sample = np.array(
                [[p.x, p.y, p.z] for p in hand],
                dtype=np.float32
            ).flatten()

            features = normalize(sample)

            distances = {
                g: score(features, prototypes[g])
                for g in GESTURES
            }

            ranked = sorted(
                distances.items(),
                key=lambda x: x[1]
            )

            prediction = ranked[0][0].upper()
            best_distance = ranked[0][1]
            second_distance = ranked[1][1]

            # Distance margin:
            # larger = more confident separation
            margin = second_distance - best_distance

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Margin: {margin:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Distance: {best_distance:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y = 155
            for g, d in ranked:
                cv2.putText(
                    frame,
                    f"{g.upper():5s}: {d:.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y += 30

        else:
            cv2.putText(
                frame,
                "Prediction: NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("V16 Step 9I - Live Validation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 70)
print("V16 STEP 9I COMPLETE")
print("=" * 70)
print(f"Frames processed : {frame_count}")
print(f"Hand frames      : {hand_frames}")
print(f"Hand coverage    : {hand_frames / frame_count * 100:.2f}%")
print(f"Average FPS      : {frame_count / (time.time() - start):.2f}")
print("=" * 70)

input("Press Enter to return to PowerShell...")
