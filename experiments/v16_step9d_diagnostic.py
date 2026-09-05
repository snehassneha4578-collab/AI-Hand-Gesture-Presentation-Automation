import cv2
import time
import traceback
import mediapipe as mp

print("=" * 70)
print("V16 STEP 9D - LIVE RECOGNIZER DIAGNOSTIC")
print("=" * 70)

MODEL_PATH = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation\models\hand_landmarker.task"

try:
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    print("Webcam opened successfully.")
    print("MediaPipe loading...")

    frame_count = 0
    hand_frames = 0
    start_time = time.time()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

        print("MediaPipe loaded successfully.")
        print()
        print("LIVE TEST STARTED")
        print("Show your hand.")
        print("Keep the window open for at least 10 seconds.")
        print("Press Q only when you want to stop.")
        print()

        while True:

            ret, frame = cap.read()

            if not ret:
                print("ERROR: Camera frame could not be read.")
                break

            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            hands = len(result.hand_landmarks)

            if hands > 0:
                hand_frames += 1
                status = "HAND DETECTED"
                display_color = (0, 255, 0)
            else:
                status = "NO HAND"
                display_color = (0, 0, 255)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Terminal diagnostic every 30 frames
            if frame_count % 30 == 0:
                print(
                    f"Frames={frame_count:4d} | "
                    f"Hands={hands} | "
                    f"HandFrames={hand_frames:4d} | "
                    f"FPS={fps:.1f}"
                )

            cv2.putText(
                frame,
                status,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                display_color,
                2
            )

            cv2.putText(
                frame,
                f"Frames: {frame_count}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Hands: {hands}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "V16 Step 9D | Q = Quit",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("V16 STEP 9D - LIVE TEST", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Q pressed.")
                break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("V16 STEP 9D COMPLETE")
    print(f"Total frames: {frame_count}")
    print(f"Frames with hand detected: {hand_frames}")
    print("=" * 70)

except Exception as e:
    print()
    print("=" * 70)
    print("ERROR DETECTED")
    print("=" * 70)
    print(str(e))
    traceback.print_exc()

finally:
    try:
        cv2.destroyAllWindows()
    except:
        pass

input("Press Enter to return to PowerShell...")
