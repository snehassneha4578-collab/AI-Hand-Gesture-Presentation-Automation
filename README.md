# AI Hand Gesture Presentation Automation

An AI-based hand gesture recognition and intelligent presentation automation system using MediaPipe, OpenCV, and personalized gesture recognition.

## Project Goal

The system aims to control presentation software through natural hand gestures, reducing the need for physical keyboards, clickers, or touch interfaces.

## Architecture

Camera
→ Hand Detection
→ 21 Hand Landmarks
→ Feature Extraction
→ Gesture Recognition
→ Confidence Estimation
→ Temporal Stability
→ Intent Engine
→ Presentation / Mouse / Annotation Actions

## Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- PyAutoGUI
- Git & GitHub

## Current Development

The project has gone through multiple recognition experiments, including:

- SVM
- Random Forest
- Extra Trees
- Geometric feature engineering
- Orientation normalization
- Personalized calibration
- Prototype-based recognition
- Confidence/rejection mechanisms
- Temporal stability
- Real-time webcam validation

The current development focus is improving reliable recognition of personalized ONE, TWO, and THREE gestures before connecting presentation-control actions.

## Current Status

Research and recognition foundation: Advanced

Real-time recognition pipeline: Working

Presentation automation: In development

Pointer and annotation features: Planned

## Disclaimer

This repository represents an actively developed academic/engineering project. Experimental models and datasets are intentionally excluded from the initial public repository when they are large or generated locally.
