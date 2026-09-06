"""
Final Evaluation Module
AI Hand Gesture Recognition & Intelligent Presentation Automation

This module consolidates measured project results.
It does not invent or recalculate experimental accuracy.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "FINAL_EVALUATION_RESULTS.txt"

def load_results():
    """Return the final evaluation report as text."""
    return RESULTS.read_text(encoding="utf-8")

def main():
    print(load_results())

if __name__ == "__main__":
    main()
