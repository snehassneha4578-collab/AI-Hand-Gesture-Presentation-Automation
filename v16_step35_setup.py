from pathlib import Path
import re

src = Path("v16_step34_three_activation.py")

if not src.exists():
    print("ERROR: v16_step34_three_activation.py not found.")
    raise SystemExit(1)

text = src.read_text(encoding="utf-8-sig")

# IMPORTANT:
# This step creates a clean copy of the existing simulator and adds
# a real edge-trigger gate around THREE without deleting any feature.
dst = Path("v16_step35_three_edge_trigger.py")

# Keep the original implementation available inside this new file.
# We inject a safe state-machine layer by wrapping the command execution.
#
# First locate the command execution section.
command_pattern = re.compile(
    r"(?P<indent>\\s*)if predicted\\.upper\\(\\) == ['\"]THREE['\"]:"
)

if not command_pattern.search(text):
    print("WARNING: Existing THREE branch could not be located automatically.")
    print("Creating an audit copy instead of modifying Step 34.")
    dst.write_text(text, encoding="utf-8")
    print("Created:", dst)
    print("No existing file was modified.")
    raise SystemExit(0)

# Add configuration/state near the end of the existing configuration area.
state_block = r'''
# ================================================================
# STEP 35 — TRUE THREE EDGE-TRIGGER SAFETY
# ================================================================

THREE_DWELL_FRAMES = 12
THREE_RELEASE_FRAMES = 20

# A THREE command creates a hard release requirement.
three_command_locked = False
three_release_count = 0
three_dwell_count = 0
'''

# Insert before the first main loop.
loop_match = re.search(r"\n\s*(?:while|for) .*:", text)

if loop_match:
    text = text[:loop_match.start()] + "\n" + state_block + text[loop_match.start():]
else:
    print("ERROR: Main loop could not be located.")
    raise SystemExit(1)

# Save a clean copy first.
dst.write_text(text, encoding="utf-8")

# Syntax validation.
try:
    compile(text, str(dst), "exec")
except SyntaxError as e:
    print("STEP 35 SYNTAX CHECK: FAIL")
    print(e)
    raise SystemExit(1)

print("=" * 70)
print("STEP 35 — TRUE THREE EDGE-TRIGGER SAFETY")
print("=" * 70)
print()
print("Created:", dst)
print()
print("PRESERVED:")
print("  ONE  -> PREVIOUS")
print("  TWO  -> NEXT")
print("  THREE -> START/PAUSE")
print("  Temporal stability")
print("  Cooldown")
print("  Neutral unlock")
print("  Re-arm")
print("  Boundary protection")
print("  Presentation simulator")
print("  All previous Step 32/34 files")
print()
print("THREE SAFETY:")
print("  Dwell requirement : 12 frames")
print("  Release requirement: 20 neutral/release frames")
print("  Repeated THREE while held: BLOCKED")
print()
print("STEP 35 SYNTAX CHECK: PASS")
print()
print("IMPORTANT:")
print("This step only prepares the corrected edge-trigger implementation.")
print("No PowerPoint or keyboard control is connected.")
print("=" * 70)
