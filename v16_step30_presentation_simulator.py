import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

ROOT = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
CSV = ROOT / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
MODEL = ROOT / "models" / "hand_landmarker.task"
REPORT = ROOT / "dataset" / "v16_step30_presentation_simulator.txt"

print("="*70)
print("STEP 30 — PRESENTATION COMMAND SIMULATOR")
print("="*70)

df = pd.read_csv(CSV)

def direction_features(row):
    pts = np.array([
        [row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]]
        for i in range(21)
    ], dtype=np.float32)

    fingers = {
        "thumb": [1,2,3,4],
        "index": [5,6,7,8],
        "middle": [9,10,11,12],
        "ring": [13,14,15,16],
        "pinky": [17,18,19,20]
    }

    feats = []

    for ids in fingers.values():
        for a, b in zip(ids[:-1], ids[1:]):
            v = pts[b] - pts[a]
            n = np.linalg.norm(v) + 1e-8
            feats.extend((v / n).tolist())

    return np.array(feats, dtype=np.float32)

def hand_to_features(hand):
    row = {}

    for i, p in enumerate(hand):
        row[f"x_{i}"] = p.x
        row[f"y_{i}"] = p.y
        row[f"z_{i}"] = p.z

    return direction_features(pd.Series(row))

X = np.array([
    direction_features(r)
    for _, r in df.iterrows()
])

centroids = {
    label: X[df["label"].values == label].mean(axis=0)
    for label in ["one", "two", "three"]
}

# ------------------------------------------------------------
# Simulator state
# ------------------------------------------------------------

TOTAL_SLIDES = 10
current_slide = 1
presentation_started = False
presentation_paused = False

STABLE_FRAMES = 7
MIN_MARGIN = 0.015
COOLDOWN = 1.0
NEUTRAL_FRAMES = 15
REARM_FRAMES = 8
REARM_FRAMES = 8

stable_prediction = None
stable_count = 0

locked = False
neutral_count = 0
rearm_count = 0
fresh_gesture_required = True
last_command_time = 0.0

command_log = []
blocked_commands = 0
blocked_commands = 0

frame_count = 0
hand_count = 0
start_time = time.time()

print("\nCOMMAND MAPPING")
print("ONE   -> PREVIOUS SLIDE")
print("TWO   -> NEXT SLIDE")
print("THREE -> START / PAUSE")
print("\nSIMULATION ONLY — NO POWERPOINT CONTROL")
print("Press Q to quit.")
print("="*70)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = int(time.time() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        prediction = None
        margin = 0.0

        if result.hand_landmarks:

            hand_count += 1

            f = hand_to_features(
                result.hand_landmarks[0]
            )

            distances = {
                label: float(
                    np.linalg.norm(f - centroid)
                )
                for label, centroid in centroids.items()
            }

            ordered = sorted(
                distances.items(),
                key=lambda x: x[1]
            )

            prediction = ordered[0][0]

            best_distance = ordered[0][1]
            second_distance = ordered[1][1]

            margin = second_distance - best_distance

            if prediction == stable_prediction:
                stable_count += 1
            else:
                stable_prediction = prediction
                stable_count = 1

            now = time.time()

            # A new command is allowed only after the previous
            # command has been followed by a genuine neutral period
            # and then a fresh gesture becomes stable.
            if not locked:
                neutral_count = 0

            if (
                not locked
                and rearm_count >= REARM_FRAMES
                and fresh_gesture_required
                and stable_count >= STABLE_FRAMES
                and margin >= MIN_MARGIN
                and now - last_command_time >= COOLDOWN
            ):

                # --------------------------------------------
                # Simulated presentation command
                # --------------------------------------------

                if prediction == "one":

                    old_slide = current_slide

                    if current_slide > 1:
                        current_slide -= 1
                        command = "PREVIOUS"
                        print(
                            f"COMMAND | ONE -> PREVIOUS | "
                            f"Slide {old_slide} -> {current_slide}"
                        )
                    else:
                        command = "PREVIOUS_BLOCKED"
                        print(
                            f"BLOCKED | ONE -> PREVIOUS | "
                            f"Already at slide {current_slide}"
                        )

                elif prediction == "two":

                    old_slide = current_slide

                    if current_slide < TOTAL_SLIDES:
                        current_slide += 1
                        command = "NEXT"
                        print(
                            f"COMMAND | TWO -> NEXT | "
                            f"Slide {old_slide} -> {current_slide}"
                        )
                    else:
                        command = "NEXT_BLOCKED"
                        print(
                            f"BLOCKED | TWO -> NEXT | "
                            f"Already at slide {current_slide}"
                        )

                else:

                    presentation_started = True
                    presentation_paused = not presentation_paused

                    command = "START/PAUSE"

                    state = (
                        "PAUSED"
                        if presentation_paused
                        else "RUNNING"
                    )

                    print(
                        f"COMMAND | THREE -> START/PAUSE | "
                        f"Presentation {state}"
                    )

                if command.endswith("_BLOCKED"):
                    blocked_commands += 1

                if command.endswith("_BLOCKED"):
                    blocked_commands += 1

                command_log.append({
                    "gesture": prediction,
                    "command": command,
                    "slide": current_slide,
                    "paused": presentation_paused,
                    "margin": margin,
                    "stable_frames": stable_count,
                    "time": now
                })

                last_command_time = now
                locked = True
                neutral_count = 0
                rearm_count = 0
                fresh_gesture_required = False
                fresh_gesture_required = False
                rearm_count = 0

        else:

            stable_prediction = None
            stable_count = 0
            neutral_count += 1

            if locked and neutral_count >= NEUTRAL_FRAMES:

                locked = False
                rearm_count = 0
                fresh_gesture_required = True

                print(
                    f"UNLOCKED | neutral={neutral_count}"
                )

            elif not locked and neutral_count >= NEUTRAL_FRAMES:

                rearm_count += 1

                if rearm_count == REARM_FRAMES:
                    print(
                        f"REARMED | neutral={neutral_count} | "
                        f"rearm={rearm_count}"
                    )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        pred_text = (
            prediction.upper()
            if prediction else
            "NO HAND"
        )

        state_text = (
            "PAUSED"
            if presentation_paused
            else
            "RUNNING"
            if presentation_started
            else
            "NOT STARTED"
        )

        lock_text = (
            "LOCKED"
            if locked
            else
            "READY"
        )

        cv2.putText(
            frame,
            f"SLIDE: {current_slide}/{TOTAL_SLIDES}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"PREDICTION: {pred_text}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"STABLE: {stable_count}/{STABLE_FRAMES}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"MARGIN: {margin:.3f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"STATE: {state_text}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"REARM: {rearm_count}/{REARM_FRAMES}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"REARM: {rearm_count}/{REARM_FRAMES}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"ENGINE: {lock_text}",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            "ONE=PREV | TWO=NEXT | THREE=START/PAUSE",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            "SIMULATION ONLY - NO POWERPOINT",
            (20, 325),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,165,255),
            2
        )

        cv2.imshow(
            "Step 30 - Presentation Simulator",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

elapsed = max(time.time() - start_time, 1e-8)

visibility = hand_count / max(frame_count, 1) * 100
fps = frame_count / elapsed

print("\n" + "="*70)
print("STEP 30 RESULTS")
print("="*70)

print(f"Frames processed : {frame_count}")
print(f"Hand frames      : {hand_count}")
print(f"Hand visibility  : {visibility:.2f}%")
print(f"Average FPS      : {fps:.2f}")

print(f"Commands executed: {len(command_log)}")
print(f"Boundary blocks   : {blocked_commands}")
print(f"Boundary blocks   : {blocked_commands}")
print(f"Final slide      : {current_slide}")
print(
    f"Presentation state: "
    f"{'PAUSED' if presentation_paused else 'RUNNING' if presentation_started else 'NOT STARTED'}"
)

if command_log:

    print("\nCommand sequence:")

    for i, item in enumerate(command_log, 1):

        print(
            f"{i:02d}. "
            f"{item['gesture'].upper()} -> "
            f"{item['command']} | "
            f"slide={item['slide']} | "
            f"margin={item['margin']:.3f}"
        )

lines = []

lines.append("="*70)
lines.append("STEP 30 — PRESENTATION COMMAND SIMULATOR")
lines.append("="*70)
lines.append("")
lines.append(f"Frames processed : {frame_count}")
lines.append(f"Hand frames      : {hand_count}")
lines.append(f"Hand visibility  : {visibility:.2f}%")
lines.append(f"Average FPS      : {fps:.2f}")
lines.append(f"Commands executed: {len(command_log)}")
lines.append(f"Boundary blocks: {blocked_commands}")
lines.append(f"Boundary blocks: {blocked_commands}")
lines.append(f"Final slide      : {current_slide}")
lines.append("")
lines.append("COMMAND MAPPING")
lines.append("ONE   -> PREVIOUS")
lines.append("TWO   -> NEXT")
lines.append("THREE -> START/PAUSE")
lines.append("")
lines.append("SAFETY")
lines.append(f"Stable frames: {STABLE_FRAMES}")
lines.append(f"Minimum margin: {MIN_MARGIN}")
lines.append(f"Cooldown: {COOLDOWN}s")
lines.append(f"Neutral reset: {NEUTRAL_FRAMES} frames")
lines.append(f"Re-arm frames: {REARM_FRAMES}")
lines.append(f"Re-arm frames: {REARM_FRAMES}")
lines.append("")
lines.append("NO POWERPOINT CONTROL WAS USED.")
lines.append("")

if command_log:

    lines.append("COMMAND LOG")

    for i, item in enumerate(command_log, 1):

        lines.append(
            f"{i:02d}. "
            f"{item['gesture'].upper()} -> "
            f"{item['command']} | "
            f"slide={item['slide']} | "
            f"margin={item['margin']:.3f}"
        )

else:

    lines.append("No commands executed.")

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\nReport saved:")
print(REPORT)

print("\n" + "="*70)
print("STEP 30 COMPLETE")
print("="*70)
