import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

ROOT = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation")
CSV = ROOT / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
MODEL = ROOT / "models" / "hand_landmarker.task"
REPORT = ROOT / "dataset" / "v16_step29_safe_command_engine.txt"

print("="*70)
print("STEP 29 — SAFE GESTURE COMMAND ENGINE")
print("="*70)

df = pd.read_csv(CSV)

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Build live-domain centroids
# ------------------------------------------------------------

X = np.array([
    direction_features(r)
    for _, r in df.iterrows()
])

centroids = {
    label: X[df["label"].values == label].mean(axis=0)
    for label in ["one", "two", "three"]
}

print("\nLive-domain centroids loaded.")

# ------------------------------------------------------------
# Safe command configuration
# ------------------------------------------------------------

STABLE_FRAMES = 5

# Step 28 showed that ONE/TWO can be correct with
# very small margins. Therefore margin is only a
# secondary safeguard.
MIN_MARGIN = 0.015

COOLDOWN = 1.0

# After accepting a command, the system must see
# this many consecutive no-hand/neutral frames
# before another command can be accepted.
NEUTRAL_FRAMES = 10

# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# State
# ------------------------------------------------------------

frame_count = 0
hand_count = 0

stable_prediction = None
stable_count = 0

locked = False
neutral_count = 0

last_command = None
last_command_time = 0.0

accepted_commands = []
rejected_low_margin = 0
rejected_unstable = 0
neutral_unlocks = 0

start = time.time()

print("\n" + "="*70)
print("SAFE COMMAND ENGINE ACTIVE")
print("="*70)
print("")
print("Gesture mapping for DRY RUN:")
print("  ONE   -> PREVIOUS")
print("  TWO   -> NEXT")
print("  THREE -> START/PAUSE")
print("")
print("NO keyboard or PowerPoint control is performed.")
print("")
print("Safety rules:")
print(f"  Stable frames : {STABLE_FRAMES}")
print(f"  Minimum margin: {MIN_MARGIN}")
print(f"  Cooldown      : {COOLDOWN}s")
print(f"  Neutral reset : {NEUTRAL_FRAMES} frames")
print("")
print("Hold gestures naturally.")
print("Press Q to quit.")
print("="*70)

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

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
        best_distance = 0.0

        # ----------------------------------------------------
        # Hand detected
        # ----------------------------------------------------

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

            # -----------------------------------------------
            # Temporal stability
            # -----------------------------------------------

            if prediction == stable_prediction:
                stable_count += 1
            else:
                stable_prediction = prediction
                stable_count = 1

            neutral_count = 0

            # -----------------------------------------------
            # Unlock only after neutral state
            # -----------------------------------------------

            if locked:
                status = "LOCKED"

            else:
                status = "READY"

            # -----------------------------------------------
            # Command acceptance
            # -----------------------------------------------

            now = time.time()

            if not locked:

                if stable_count >= STABLE_FRAMES:

                    if margin >= MIN_MARGIN:

                        if (
                            now - last_command_time
                            >= COOLDOWN
                        ):

                            command = {
                                "one": "PREVIOUS",
                                "two": "NEXT",
                                "three": "START/PAUSE"
                            }[prediction]

                            accepted_commands.append({
                                "gesture": prediction,
                                "command": command,
                                "margin": margin,
                                "stable_frames": stable_count,
                                "timestamp": now
                            })

                            last_command = command
                            last_command_time = now

                            locked = True
                            neutral_count = 0

                            print(
                                f"ACCEPTED | "
                                f"{prediction.upper():5s} -> "
                                f"{command:10s} | "
                                f"margin={margin:.3f} | "
                                f"stable={stable_count}"
                            )

                    else:

                        rejected_low_margin += 1

                else:

                    rejected_unstable += 1

        # ----------------------------------------------------
        # No hand = neutral state
        # ----------------------------------------------------

        else:

            stable_prediction = None
            stable_count = 0

            neutral_count += 1

            prediction = None
            margin = 0.0

            if locked and neutral_count >= NEUTRAL_FRAMES:

                locked = False
                neutral_unlocks += 1

                print(
                    f"UNLOCKED | neutral frames="
                    f"{neutral_count}"
                )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        if prediction:

            display_prediction = prediction.upper()

        else:

            display_prediction = "NO HAND"

        state = "LOCKED" if locked else "READY"

        cv2.putText(
            frame,
            f"PREDICTION: {display_prediction}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"MARGIN: {margin:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"STABLE: {stable_count}/{STABLE_FRAMES}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"NEUTRAL: {neutral_count}/{NEUTRAL_FRAMES}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"COMMANDS: {len(accepted_commands)}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            "ONE=PREVIOUS | TWO=NEXT | THREE=START/PAUSE",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255,255,0),
            2
        )

        cv2.putText(
            frame,
            "DRY RUN ONLY - NO POWERPOINT CONTROL",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,165,255),
            2
        )

        cv2.imshow(
            "Step 29 - Safe Gesture Command Engine",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------
# Final statistics
# ------------------------------------------------------------

elapsed = max(time.time() - start, 1e-8)

visibility = (
    hand_count / max(frame_count, 1) * 100
)

fps = frame_count / elapsed

print("\n" + "="*70)
print("STEP 29 RESULTS")
print("="*70)

print(f"Frames processed : {frame_count}")
print(f"Hand frames      : {hand_count}")
print(f"Hand visibility  : {visibility:.2f}%")
print(f"Average FPS      : {fps:.2f}")

print(
    f"Accepted commands: "
    f"{len(accepted_commands)}"
)

print(
    f"Low-margin rejections: "
    f"{rejected_low_margin}"
)

print(
    f"Unstable rejections: "
    f"{rejected_unstable}"
)

print(
    f"Neutral unlocks: "
    f"{neutral_unlocks}"
)

if accepted_commands:

    print("\nAccepted command sequence:")

    for i, item in enumerate(
        accepted_commands,
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{item['gesture'].upper():5s} -> "
            f"{item['command']:10s} | "
            f"margin={item['margin']:.3f} | "
            f"stable={item['stable_frames']}"
        )

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

lines = []

lines.append("="*70)
lines.append("STEP 29 — SAFE GESTURE COMMAND ENGINE")
lines.append("="*70)
lines.append("")

lines.append(
    f"Frames processed : {frame_count}"
)

lines.append(
    f"Hand frames      : {hand_count}"
)

lines.append(
    f"Hand visibility  : {visibility:.2f}%"
)

lines.append(
    f"Average FPS      : {fps:.2f}"
)

lines.append(
    f"Accepted commands: {len(accepted_commands)}"
)

lines.append(
    f"Low-margin rejections: {rejected_low_margin}"
)

lines.append(
    f"Unstable rejections: {rejected_unstable}"
)

lines.append(
    f"Neutral unlocks: {neutral_unlocks}"
)

lines.append("")

lines.append("COMMAND MAPPING")
lines.append("ONE   -> PREVIOUS")
lines.append("TWO   -> NEXT")
lines.append("THREE -> START/PAUSE")

lines.append("")

lines.append("SAFETY MECHANISMS")
lines.append(
    f"Stable frames required: {STABLE_FRAMES}"
)
lines.append(
    f"Minimum margin: {MIN_MARGIN}"
)
lines.append(
    f"Cooldown: {COOLDOWN}s"
)
lines.append(
    f"Neutral frames required for unlock: "
    f"{NEUTRAL_FRAMES}"
)

lines.append("")

if accepted_commands:

    lines.append("ACCEPTED COMMANDS")

    for i, item in enumerate(
        accepted_commands,
        start=1
    ):

        lines.append(
            f"{i:02d}. "
            f"{item['gesture'].upper()} -> "
            f"{item['command']} | "
            f"margin={item['margin']:.3f} | "
            f"stable={item['stable_frames']}"
        )

else:

    lines.append(
        "No commands were accepted during this run."
    )

lines.append("")
lines.append(
    "IMPORTANT: This was a DRY RUN."
)
lines.append(
    "No keyboard, mouse, or PowerPoint action was performed."
)

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("\nReport saved:")
print(REPORT)

print("\n" + "="*70)
print("STEP 29 COMPLETE")
print("="*70)
