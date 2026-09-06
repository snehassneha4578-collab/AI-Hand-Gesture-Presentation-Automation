from pathlib import Path

# STEP 34 THREE ACTIVATION SAFETY
THREE_STABLE_FRAMES = 12
THREE_MIN_MARGIN = 0.10
THREE_RELEASE_FRAMES = 20
three_release_counter = THREE_RELEASE_FRAMES


p = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation\v16_step30_presentation_simulator.py")
s = p.read_text(encoding="utf-8-sig")

# Stronger command-cycle settings
s = s.replace(
    "STABLE_FRAMES = 5",
    "STABLE_FRAMES = 7"
)

s = s.replace(
    "NEUTRAL_FRAMES = 10",
    "NEUTRAL_FRAMES = 15"
)

s = s.replace(
    "REARM_FRAMES = 5",
    "REARM_FRAMES = 8"
)

# Add gesture-release tracking
s = s.replace(
    "rearm_count = 0\nlast_command_time = 0.0",
    "rearm_count = 0\nfresh_gesture_required = True\nlast_command_time = 0.0"
)

# Require a fresh gesture after re-arm
s = s.replace(
    """if (
                not locked
                and rearm_count >= REARM_FRAMES
                and stable_count >= STABLE_FRAMES
                and margin >= MIN_MARGIN
                and now - last_command_time >= COOLDOWN
            ):""",
    """if (
                not locked
                and rearm_count >= REARM_FRAMES
                and fresh_gesture_required
                and stable_count >= STABLE_FRAMES
                and margin >= MIN_MARGIN
                and now - last_command_time >= COOLDOWN
            ):"""
)

# Mark gesture as consumed after command
s = s.replace(
    """last_command_time = now
                locked = True
                neutral_count = 0
                rearm_count = 0""",
    """last_command_time = now
                locked = True
                neutral_count = 0
                rearm_count = 0
                fresh_gesture_required = False"""
)

# When genuine neutral period is reached, allow a new gesture
s = s.replace(
    """if locked and neutral_count >= NEUTRAL_FRAMES:

                locked = False
                rearm_count = 0

                print(
                    f"UNLOCKED | neutral={neutral_count}"
                )""",
    """if locked and neutral_count >= NEUTRAL_FRAMES:

                locked = False
                rearm_count = 0
                fresh_gesture_required = True

                print(
                    f"UNLOCKED | neutral={neutral_count}"
                )"""
)

# Prevent counting boundary commands as executed commands
s = s.replace(
    """if current_slide > 1:
                        current_slide -= 1

                    command = "PREVIOUS"

                    print(
                        f"COMMAND | ONE -> PREVIOUS | "
                        f"Slide {old_slide} -> {current_slide}"
                    )""",
    """if current_slide > 1:
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
                        )"""
)

s = s.replace(
    """if current_slide < TOTAL_SLIDES:
                        current_slide += 1

                    command = "NEXT"

                    print(
                        f"COMMAND | TWO -> NEXT | "
                        f"Slide {old_slide} -> {current_slide}"
                    )""",
    """if current_slide < TOTAL_SLIDES:
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
                        )"""
)

# Add blocked counter
s = s.replace(
    "command_log = []",
    "command_log = []\nblocked_commands = 0"
)

# Count boundary blocks
s = s.replace(
    """command_log.append({
                    "gesture": prediction,""",
    """if command.endswith("_BLOCKED"):
                    blocked_commands += 1

                command_log.append({
                    "gesture": prediction,"""
)

# Report blocked attempts
s = s.replace(
    'lines.append(f"Commands executed: {len(command_log)}")',
    'lines.append(f"Commands executed: {len(command_log)}")\nlines.append(f"Boundary blocks: {blocked_commands}")'
)

# Console summary
s = s.replace(
    'print(f"Commands executed: {len(command_log)}")',
    'print(f"Commands executed: {len(command_log)}")\nprint(f"Boundary blocks   : {blocked_commands}")'
)

p.write_text(s, encoding="utf-8")

print("=" * 70)
print("STEP 32 — SAFER GESTURE ACTIVATION")
print("=" * 70)
print("STABLE_FRAMES   = 7")
print("NEUTRAL_FRAMES  = 15")
print("REARM_FRAMES    = 8")
print("Simulation only — NO PowerPoint control")
print("Press Q to quit.")
print("=" * 70)

exec(
    compile(
        s,
        str(p),
        "exec"
    ),
    {"__name__": "__main__"}
)
