from pathlib import Path

p = Path(r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation\v16_step30_presentation_simulator.py")

data = p.read_text(encoding="utf-8-sig")

p.write_text(data, encoding="utf-8")

print("BOM removed successfully.")
print("Starting Step 31...")

