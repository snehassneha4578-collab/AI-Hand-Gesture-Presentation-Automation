import cv2
import sys
import platform

print("=" * 70)
print("STEP 33 — OPENCV 5 COMPETITION COMPLIANCE AUDIT")
print("=" * 70)

print(f"Python version  : {sys.version.split()[0]}")
print(f"OpenCV version  : {cv2.__version__}")
print(f"Platform        : {platform.system()} {platform.machine()}")

major = int(cv2.__version__.split(".")[0])

print()
print("OPEN-CV 5 REQUIREMENT")
print("-" * 70)

if major >= 5:
    print("PASS: OpenCV 5+ detected")
else:
    print("FAIL: OpenCV 5 is required")

print()
print("OPEN-CV VISION CAPABILITIES")
print("-" * 70)

checks = {
    "VideoCapture": hasattr(cv2, "VideoCapture"),
    "VideoWriter": hasattr(cv2, "VideoWriter"),
    "resize": hasattr(cv2, "resize"),
    "cvtColor": hasattr(cv2, "cvtColor"),
    "GaussianBlur": hasattr(cv2, "GaussianBlur"),
    "Canny": hasattr(cv2, "Canny"),
    "threshold": hasattr(cv2, "threshold"),
    "findContours": hasattr(cv2, "findContours"),
    "rectangle": hasattr(cv2, "rectangle"),
    "putText": hasattr(cv2, "putText"),
}

passed = 0

for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"{status:6} {name}")
    if result:
        passed += 1

print()
print(f"Capability checks: {passed}/{len(checks)}")

print()
print("CURRENT PROJECT PIPELINE")
print("-" * 70)

pipeline = [
    "Webcam video acquisition",
    "OpenCV frame processing",
    "Real-time hand/visual analysis",
    "MediaPipe 21-landmark extraction",
    "AI gesture classification",
    "Temporal stability filtering",
    "False-trigger prevention",
    "Presentation intent generation",
]

for i, item in enumerate(pipeline, 1):
    print(f"{i}. {item}")

print()
print("=" * 70)

if major >= 5 and passed == len(checks):
    print("STEP 33 STATUS: READY")
    print("OpenCV 5 is available for the competition pipeline.")
else:
    print("STEP 33 STATUS: NEEDS FIX")

print("=" * 70)
