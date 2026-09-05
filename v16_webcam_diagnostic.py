import cv2
import time

print("=" * 70)
print("V16 STEP 9 DIAGNOSTIC - REAL-TIME WEBCAM TEST")
print("=" * 70)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    input("Press Enter to exit...")
    raise SystemExit

print("Webcam opened successfully.")
print("Show your hand to the camera.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    cv2.putText(
        frame,
        "WEBCAM WORKING - Press Q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("V16 Webcam Diagnostic", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("=" * 70)
print("WEBCAM DIAGNOSTIC COMPLETE")
print("=" * 70)
