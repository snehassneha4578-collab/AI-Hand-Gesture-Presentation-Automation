from pathlib import Path

p = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation\v16_step30_presentation_simulator.py")

s = p.read_text(encoding="utf-8")

s = s.replace(
'NEUTRAL_FRAMES = 10',
'NEUTRAL_FRAMES = 10\nREARM_FRAMES = 5'
)

s = s.replace(
'neutral_count = 0\nlast_command_time = 0.0',
'neutral_count = 0\nrearm_count = 0\nlast_command_time = 0.0'
)

s = s.replace(
'''            neutral_count = 0

            now = time.time()

            if (
                not locked
                and stable_count >= STABLE_FRAMES
                and margin >= MIN_MARGIN
                and now - last_command_time >= COOLDOWN
            ):''',
'''            now = time.time()

            # A new command is allowed only after the previous
            # command has been followed by a genuine neutral period
            # and then a fresh gesture becomes stable.
            if not locked:
                neutral_count = 0

            if (
                not locked
                and rearm_count >= REARM_FRAMES
                and stable_count >= STABLE_FRAMES
                and margin >= MIN_MARGIN
                and now - last_command_time >= COOLDOWN
            ):'''
)

s = s.replace(
'''                locked = True
                neutral_count = 0''',
'''                locked = True
                neutral_count = 0
                rearm_count = 0'''
)

s = s.replace(
'''            stable_prediction = None
            stable_count = 0
            neutral_count += 1

            if locked and neutral_count >= NEUTRAL_FRAMES:

                locked = False

                print(
                    f"UNLOCKED | neutral={neutral_count}"
                )''',
'''            stable_prediction = None
            stable_count = 0
            neutral_count += 1

            if locked and neutral_count >= NEUTRAL_FRAMES:

                locked = False
                rearm_count = 0

                print(
                    f"UNLOCKED | neutral={neutral_count}"
                )

            elif not locked and neutral_count >= NEUTRAL_FRAMES:

                rearm_count += 1

                if rearm_count == REARM_FRAMES:
                    print(
                        f"REARMED | neutral={neutral_count} | "
                        f"rearm={rearm_count}"
                    )'''
)

# Add rearm display
s = s.replace(
'''        cv2.putText(
            frame,
            f"ENGINE: {lock_text}",''',
'''        cv2.putText(
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
            f"ENGINE: {lock_text}",'''
)

# Move mapping text lower to avoid overlap
s = s.replace(
'''            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,''',
'''            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,''',
1
)

s = s.replace(
'''            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,165,255),''',
'''            (20, 325),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,165,255),'''
)

# Add configuration to report
s = s.replace(
'''lines.append(f"Neutral reset: {NEUTRAL_FRAMES} frames")''',
'''lines.append(f"Neutral reset: {NEUTRAL_FRAMES} frames")
lines.append(f"Re-arm frames: {REARM_FRAMES}")'''
)

p.write_text(s, encoding="utf-8")

print("Step 31 command-cycle engine updated.")
print("Starting Step 31 simulator...")
print("Press Q to quit.")

exec(
    compile(
        s,
        str(p),
        "exec"
    ),
    {"__name__": "__main__"}
)
