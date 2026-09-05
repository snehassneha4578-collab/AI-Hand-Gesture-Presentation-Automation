import cv2
import time
import mediapipe as mp

print("=" * 70)
print("V16 STEP 9B - MEDIAPIPE REAL-TIME TEST")
print("=" * 70)

MODEL_PATH = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation\models\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam failed.")
    input("Press Enter...")
    raise SystemExit

print("Webcam opened.")
print("MediaPipe Hand Landmarker loading...")
print("Press Q to quit.")

frame_count = 0
last_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Frame read failed.")
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        hand_count = len(result.hand_landmarks)

        if hand_count > 0:
            status = "HAND DETECTED"
            text_color = (0, 255, 0)
        else:
            status = "NO HAND"
            text_color = (0, 0, 255)

        fps = 1.0 / max(time.time() - last_time, 0.001)
        last_time = time.time()

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            2
        )

        cv2.putText(
            frame,
            f"Hands: {hand_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {frame_count}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("V16 Step 9B - MediaPipe Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Q pressed - closing.")
            break

cap.release()
cv2.destroyAllWindows()

print("=" * 70)
print("V16 STEP 9B COMPLETE")
print(f"Total frames processed: {frame_count}")
print("=" * 70)

input("Press Enter to return to PowerShell...")
