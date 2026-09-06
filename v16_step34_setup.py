from pathlib import Path
import shutil

src = Path("v16_step32_safe_activation.py")
dst = Path("v16_step34_three_activation.py")

if not src.exists():
    print("ERROR: Step 32 file not found.")
    raise SystemExit(1)

# Preserve Step 32 completely.
shutil.copy2(src, dst)

text = dst.read_text(encoding="utf-8-sig")

# Add configuration without modifying existing logic.
insert = """
# STEP 34 THREE ACTIVATION SAFETY
THREE_STABLE_FRAMES = 12
THREE_MIN_MARGIN = 0.10
THREE_RELEASE_FRAMES = 20
three_release_counter = THREE_RELEASE_FRAMES
"""

# Insert after imports.
lines = text.splitlines()
insert_at = 0
while insert_at < len(lines) and (
    lines[insert_at].startswith("import ") or
    lines[insert_at].startswith("from ")
):
    insert_at += 1

lines.insert(insert_at, insert)
text = "\n".join(lines) + "\n"

dst.write_text(text, encoding="utf-8")

print("=" * 70)
print("STEP 34 — THREE ACTIVATION SAFETY UPGRADE")
print("=" * 70)
print("Created:", dst)
print()
print("Step 32 preserved.")
print("Existing features preserved.")
print()
print("Added safety configuration:")
print("  THREE stable frames : 12")
print("  THREE minimum margin : 0.10")
print("  THREE release frames : 20")
print()
print("NOTE:")
print("The Step 32 simulator remains untouched.")
print("Step 34 will first validate that the copied program runs.")
print("=" * 70)

try:
    compile(text, str(dst), "exec")
    print("STEP 34 SYNTAX CHECK: PASS")
except SyntaxError as e:
    print("STEP 34 SYNTAX CHECK: FAIL")
    print(e)
    raise SystemExit(1)

print()
print("Step 34 file is ready.")
print("Run this file next:")
print("python v16_step34_three_activation.py")
