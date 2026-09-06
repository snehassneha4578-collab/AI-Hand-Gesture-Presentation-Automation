import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
import argparse
import pythoncom
import win32com.client
import pyautogui


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
MODEL = ROOT / "models" / "hand_landmarker.task"
PPTX = ROOT / "gesture_controller_test_10_slides.pptx"

STABLE_FRAMES = 7
ONE_MIN_MARGIN = 0.08
TWO_MIN_MARGIN = 0.08
THREE_MIN_MARGIN = 0.15
COOLDOWN = 1.0
NEUTRAL_FRAMES = 15
REARM_FRAMES = 8
THREE_RELEASE_FRAMES = 20


# ============================================================
# FEATURE EXTRACTION
# ============================================================

df = pd.read_csv(CSV)


def direction_features(row):
    pts = np.array([
        [row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]]
        for i in range(21)
    ], dtype=np.float32)

    fingers = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
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


# ============================================================
# POWERPOINT
# ============================================================

class PowerPointController:

    def __init__(self, pptx_path, live=False):

        self.live = live
        self.app = None
        self.presentation = None

        if not self.live:
            print("POWERPOINT | DRY-RUN MODE")
            print("POWERPOINT | NO COMMANDS WILL BE SENT")
            return

        pythoncom.CoInitialize()

        try:
            self.app = win32com.client.GetActiveObject(
                "PowerPoint.Application"
            )
            print("POWERPOINT | Connected to existing instance")

        except Exception:

            self.app = win32com.client.Dispatch(
                "PowerPoint.Application"
            )

            self.app.Visible = True
            print("POWERPOINT | Started PowerPoint")

        if self.app.Presentations.Count == 0:

            if not pptx_path.exists():
                raise FileNotFoundError(
                    f"Presentation not found: {pptx_path}"
                )

            self.presentation = self.app.Presentations.Open(
                str(pptx_path),
                WithWindow=True
            )

            print(
                f"POWERPOINT | Opened: {pptx_path.name}"
            )

        else:

            self.presentation = self.app.Presentations(1)

            print(
                f"POWERPOINT | Using: "
                f"{self.presentation.Name}"
            )


    def get_view(self):

        if not self.live:
            return None

        if self.app.SlideShowWindows.Count == 0:
            return None

        return self.app.SlideShowWindows(1).View


    def start(self):

        if not self.live:
            print("DRY-RUN | THREE -> START")
            return True

        view = self.get_view()

        if view is not None:
            return True

        view = self.presentation.SlideShowSettings.Run()

        time.sleep(0.8)

        print("POWERPOINT | THREE -> START SLIDESHOW")

        return True


    def exit_slideshow(self):

        if not self.live:
            print("DRY-RUN | THREE -> EXIT")
            return True

        view = self.get_view()

        if view is None:
            print("POWERPOINT | EXIT blocked - slideshow not running")
            return False

        try:
            view.Exit()
            time.sleep(0.5)

            print("POWERPOINT | THREE -> EXIT SLIDESHOW")

            return True

        except Exception as e:

            print(
                f"POWERPOINT | EXIT ERROR: {e}"
            )

            return False


    def previous(self):

        if not self.live:
            print("DRY-RUN | ONE -> PREVIOUS")
            return True

        view = self.get_view()

        if view is None:
            print(
                "POWERPOINT | PREVIOUS blocked - "
                "slideshow not running"
            )
            return False

        current = int(view.CurrentShowPosition)

        if current <= 1:
            print(
                f"POWERPOINT | PREVIOUS blocked | "
                f"Already at slide {current}"
            )
            return False

        view.Previous()

        new_slide = int(view.CurrentShowPosition)

        print(
            f"POWERPOINT | ONE -> PREVIOUS | "
            f"Slide {current} -> {new_slide}"
        )

        return True


    def next(self):

        if not self.live:
            print("DRY-RUN | TWO -> NEXT")
            return True

        view = self.get_view()

        if view is None:
            print(
                "POWERPOINT | NEXT blocked - "
                "slideshow not running"
            )
            return False

        total = int(self.presentation.Slides.Count)
        current = int(view.CurrentShowPosition)

        if current >= total:
            print(
                f"POWERPOINT | NEXT blocked | "
                f"Already at slide {current}"
            )
            return False

        view.Next()

        new_slide = int(view.CurrentShowPosition)

        print(
            f"POWERPOINT | TWO -> NEXT | "
            f"Slide {current} -> {new_slide}"
        )

        return True


    def three(self):

        if not self.live:
            print("DRY-RUN | THREE -> START/EXIT")
            return True

        view = self.get_view()

        if view is None:
            return self.start()

        return self.exit_slideshow()


    def status(self):

        if not self.live:
            return "DRY-RUN", 0, 10

        view = self.get_view()

        if view is None:
            return (
                "NOT STARTED",
                0,
                int(self.presentation.Slides.Count)
            )

        slide = int(view.CurrentShowPosition)
        total = int(self.presentation.Slides.Count)

        try:
            state = int(view.State)
        except Exception:
            state = 1

        if state == 2:
            state_text = "PAUSED"
        elif state in (3, 4):
            state_text = "SCREEN"
        else:
            state_text = "RUNNING"

        return state_text, slide, total


# ============================================================
# SAFE COMMAND ENGINE
# ============================================================

class SafeCommandEngine:

    def __init__(self, ppt):

        self.ppt = ppt

        self.locked = False
        self.neutral_count = 0
        self.rearm_count = 0

        self.fresh_gesture_required = True

        self.stable_prediction = None
        self.stable_count = 0

        self.last_command_time = 0.0

        self.three_armed = True
        self.three_release_count = 0

        self.command_count = 0
        self.blocked_count = 0


    def process(self, prediction, margin):

        now = time.time()

        if prediction is not None:

            if prediction == self.stable_prediction:
                self.stable_count += 1
            else:
                self.stable_prediction = prediction
                self.stable_count = 1

            if prediction == "three":

                self.three_release_count = 0

            else:

                self.three_release_count += 1

                if (
                    self.three_release_count >= THREE_RELEASE_FRAMES
                    and not self.locked
                ):
                    self.three_armed = True

            threshold = {
                "one": ONE_MIN_MARGIN,
                "two": TWO_MIN_MARGIN,
                "three": THREE_MIN_MARGIN
            }.get(prediction, 999.0)

            safe = (
                not self.locked
                and self.rearm_count >= REARM_FRAMES
                and self.fresh_gesture_required
                and self.stable_count >= STABLE_FRAMES
                and margin >= threshold
                and now - self.last_command_time >= COOLDOWN
                and (
                    prediction != "three"
                    or self.three_armed
                )
            )

            if safe:

                if prediction == "one":
                    success = self.ppt.previous()
                    command = "PREVIOUS"

                elif prediction == "two":
                    success = self.ppt.next()
                    command = "NEXT"

                else:
                    success = self.ppt.three()
                    command = "START/EXIT"

                if success:
                    self.command_count += 1
                else:
                    self.blocked_count += 1

                result = "SUCCESS" if success else "BLOCKED/FAILED"

                print(
                    f"COMMAND | {prediction.upper()} -> "
                    f"{command} | "
                    f"margin={margin:.3f} | "
                    f"result={result}"
                )

                self.last_command_time = now

                if prediction == "three":
                    self.three_armed = False
                    self.three_release_count = 0

                self.locked = True
                self.neutral_count = 0
                self.rearm_count = 0
                self.fresh_gesture_required = False

                return command

        else:

            self.stable_prediction = None
            self.stable_count = 0
            self.neutral_count += 1

            if (
                self.locked
                and self.neutral_count >= NEUTRAL_FRAMES
            ):

                self.locked = False
                self.rearm_count = 0
                self.fresh_gesture_required = True

                print(
                    f"UNLOCKED | neutral={self.neutral_count}"
                )

            elif (
                not self.locked
                and self.neutral_count >= NEUTRAL_FRAMES
            ):

                self.rearm_count += 1

                if self.rearm_count == REARM_FRAMES:
                    print(
                        f"REARMED | neutral={self.neutral_count} | "
                        f"rearm={self.rearm_count}"
                    )

        return None


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

mode = parser.add_mutually_exclusive_group()

mode.add_argument(
    "--dry-run",
    action="store_true"
)

mode.add_argument(
    "--live",
    action="store_true"
)

args = parser.parse_args()

live_mode = args.live


print("=" * 70)
print("POWERPOINT GESTURE CONTROLLER V2")
print("=" * 70)

print("ONE   -> PREVIOUS SLIDE")
print("TWO   -> NEXT SLIDE")
print("THREE -> START / EXIT SLIDESHOW")

if live_mode:
    print("MODE  -> LIVE POWERPOINT CONTROL")
else:
    print("MODE  -> DRY RUN")
    print("NO POWERPOINT COMMANDS WILL BE SENT")

print("Press Q to quit.")
print("=" * 70)


# ============================================================
# CONTROLLER
# ============================================================

ppt = PowerPointController(
    PPTX,
    live=live_mode
)

engine = SafeCommandEngine(ppt)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)


cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not cap.isOpened():
    raise RuntimeError(
        "Camera could not be opened"
    )


frame_count = 0
hand_count = 0
start_time = time.time()


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        frame_count += 1

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        timestamp = int(
            time.time() * 1000
        )

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
                    np.linalg.norm(
                        f - centroid
                    )
                )
                for label, centroid
                in centroids.items()
            }

            ordered = sorted(
                distances.items(),
                key=lambda x: x[1]
            )

            prediction = ordered[0][0]

            margin = (
                ordered[1][1]
                - ordered[0][1]
            )

            engine.process(
                prediction,
                margin
            )

        else:

            engine.process(
                None,
                0.0
            )


        state, slide, total = ppt.status()

        fps = (
            frame_count
            / max(
                time.time() - start_time,
                1e-6
            )
        )

        visibility = (
            hand_count
            / max(frame_count, 1)
            * 100
        )


        pred_text = (
            prediction.upper()
            if prediction
            else "NO HAND"
        )

        lock_text = (
            "LOCKED"
            if engine.locked
            else "READY"
        )


        cv2.putText(
            frame,
            f"SLIDE: {slide}/{total}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"PREDICTION: {pred_text}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"STABLE: {engine.stable_count}/{STABLE_FRAMES}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MARGIN: {margin:.3f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"LOCK: {lock_text}",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"REARM: {engine.rearm_count}/{REARM_FRAMES}",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"THREE: "
                f"{'ARMED' if engine.three_armed else 'WAIT'} "
                f"{engine.three_release_count}/"
                f"{THREE_RELEASE_FRAMES}"
            ),
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 315),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"VISIBILITY: {visibility:.1f}%",
            (20, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "PowerPoint Gesture Controller V2",
            frame
        )


        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break


cap.release()
cv2.destroyAllWindows()

print("=" * 70)
print("POWERPOINT GESTURE CONTROLLER V2 STOPPED")
print("=" * 70)

print(f"Frames: {frame_count}")
print(f"Hand frames: {hand_count}")

print(
    f"Visibility: "
    f"{hand_count / max(frame_count, 1) * 100:.2f}%"
)

print(
    f"Average FPS: "
    f"{frame_count / max(time.time() - start_time, 1e-6):.2f}"
)

print(
    f"Commands: {engine.command_count}"
)

print(
    f"Blocked/failed: {engine.blocked_count}"
)

if live_mode:
    pythoncom.CoUninitialize()

