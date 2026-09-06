# AI-Based Hand Gesture Recognition & Intelligent Presentation Automation System

## 1. Project Overview

The **AI-Based Hand Gesture Recognition & Intelligent Presentation Automation System** is a real-time computer-vision system that enables users to control presentation software through natural hand gestures.

The system replaces conventional keyboard, mouse, and presentation-clicker interactions with a contactless gesture-based interface.

The implemented prototype uses **OpenCV 5** for real-time video processing, **MediaPipe Hand Landmarker** for 21-point hand landmark extraction, personalized live-domain gesture recognition, a safety-focused command engine, Microsoft PowerPoint automation, and **Amazon DynamoDB** for cloud-based session analytics.

### Primary Gesture Mapping

| Gesture | Action                 |
| ------- | ---------------------- |
| ONE     | Previous Slide         |
| TWO     | Next Slide             |
| THREE   | Start / Exit Slideshow |

---

## 2. Problem Statement

Traditional presentation systems require physical interaction with keyboards, mice, or presentation remotes.

This can interrupt the presenter's flow and is inconvenient when the presenter wants to remain away from the computer.

This project addresses the problem through a contactless interface in which hand gestures are detected, recognized, validated through safety logic, and converted into presentation commands.

---

## 3. Target Users

The system is intended for:

* Students and teachers
* Technical presenters
* Conference speakers
* Demonstrators
* Trainers
* Users requiring a contactless presentation interface

The prototype is particularly suitable for controlled presentation environments where the camera can clearly observe the user's hand.

---

## 4. System Architecture

The implemented processing pipeline is:

**Webcam**

↓

**OpenCV 5 Video Processing**

↓

**MediaPipe Hand Landmarker**

↓

**21 Hand Landmarks**

↓

**Feature Extraction**

↓

**Gesture Recognition**

↓

**Confidence / Margin Estimation**

↓

**Temporal Stability & Safety Engine**

↓

**Intent Engine**

↓

**PowerPoint Automation**

↓

**Session Analytics & Evaluation**

↓

**AWS DynamoDB**

### 4.1 Webcam Input

The webcam provides continuous video frames to the real-time vision pipeline.

### 4.2 OpenCV 5

OpenCV 5 performs the substantive image/video processing stage, including:

* Webcam capture
* Frame processing
* Image conversion
* Real-time visualization
* FPS measurement
* Dashboard rendering

The validated environment uses:

**OpenCV 5.0.0**

### 4.3 MediaPipe Hand Landmarker

MediaPipe Hand Landmarker detects the user's hand and extracts 21 hand landmarks.

Each landmark contains X, Y, and Z coordinates:

**21 × 3 = 63 raw landmark features**

### 4.4 Feature Extraction

The system uses raw landmark coordinates together with engineered geometric and direction-related features.

Multiple feature representations were investigated during development, including:

* Raw landmark features
* Relative positions
* Geometric features
* Direction features
* Orientation normalization
* Finger-state analysis
* Personalized prototypes

### 4.5 Gesture Recognition

Multiple recognition approaches were evaluated during development:

* Support Vector Machine
* Random Forest
* Extra Trees
* Prototype-based recognition
* Hierarchical recognition
* Personalized/live-domain calibration

The final real-time controller uses **live-domain gesture prototypes and direction-related features**, combined with safety and temporal decision logic.

### 4.6 Safety Engine

Because an incorrect gesture could cause an unintended presentation action, the system uses multiple layers of command protection.

Implemented mechanisms include:

* Temporal stability filtering
* Prediction-margin filtering
* Neutral-state unlocking
* Re-arm requirement
* Fresh-gesture requirement
* Cooldown protection
* THREE edge-trigger protection
* Release detection
* Presentation boundary protection
* False-trigger prevention

The system therefore does not execute a presentation action from a single unstable frame.

---

## 5. Intent Engine

The recognized gesture is converted into an application-level intent.

| Recognized Gesture | Intent     | PowerPoint Action       |
| ------------------ | ---------- | ----------------------- |
| ONE                | PREVIOUS   | Move to previous slide  |
| TWO                | NEXT       | Move to next slide      |
| THREE              | START/EXIT | Start or exit slideshow |

PowerPoint automation was directly tested using **Microsoft PowerPoint COM automation**.

Previous, next, start, and exit operations were successfully verified.

The controller also rejects commands when the slideshow is not running or when presentation boundaries prevent the requested action.

---

## 6. Real-Time Performance

A live-domain real-time test processed:

* Frames: **2,079**
* Hand frames: **1,739**
* Hand visibility: **83.65%**
* Average FPS: **25.86**

This demonstrates real-time operation of the vision pipeline on the development system.

A separate controlled safety/presentation test achieved:

* Hand visibility: **91.73%**
* Average FPS: **29.58**

Performance can vary depending on camera conditions, hand visibility, lighting, and system load.

---

## 7. Recognition Evaluation

The strongest controlled live-calibration evaluation produced:

| Gesture     |   Correct |   Accuracy |
| ----------- | --------: | ---------: |
| ONE         |     10/10 |    100.00% |
| TWO         |     10/10 |    100.00% |
| THREE       |     11/13 |     84.62% |
| **Overall** | **31/33** | **93.94%** |

### Important Qualification

The **93.94% result is a controlled real-time recognition measurement using live calibration/prototypes**.

It is **not an unseen-user or universal accuracy claim**.

This distinction is important because the experiments demonstrated measurable differences between calibration data and live operating data.

The result should therefore be interpreted as evidence of the prototype's recognition capability under controlled calibrated conditions.

---

## 8. Domain-Shift Analysis

The project explicitly investigated differences between calibration data and live operating data.

Observed results:

* Average feature shift: **1.2486 SD**
* Features exceeding 2 SD: **8/45**
* Features exceeding 3 SD: **2/45**
* Diagnosis: **Moderate domain shift**

The TWO gesture exhibited the largest class-specific feature shift.

This analysis demonstrates that recognition performance can change when operating conditions differ from those used during calibration.

---

## 9. Domain Adaptation

Live-domain adaptation was evaluated to determine whether adapting gesture prototypes to the current operating environment improves recognition.

| Method                         | Accuracy |
| ------------------------------ | -------: |
| Original calibration centroids |   34.00% |
| 50/50 adapted centroids        |   49.33% |
| Pure live centroids            |   82.00% |

The 50/50 adaptation produced an improvement of:

**+15.33 percentage points**

These results support the use of live/personal calibration while also demonstrating that generalized recognition remains a limitation.

The pure live-centroid result represents **within-session performance** and is not treated as unseen-user generalization.

---

## 10. AWS Cloud Component

AWS is used as a meaningful cloud component for session analytics.

### AWS Service

**Amazon DynamoDB**

### Table

**GestureSessionAnalytics**

### Region

**ap-southeast-2**

### Stored Information

Example session analytics include:

* Session ID
* Timestamp
* OpenCV version
* FPS
* Hand visibility
* Controlled live accuracy
* Successful commands
* Blocked/failed commands
* Safety status
* AWS connection status

A real project evaluation record was successfully written to and verified from DynamoDB.

AWS credentials are maintained outside the source repository and are not stored in project files.

---

## 11. Safety and False-Trigger Prevention

Presentation automation requires more than gesture classification because an incorrect prediction can cause an unwanted action.

The safety architecture therefore uses a state-based command engine.

A command must satisfy appropriate recognition and safety conditions before execution.

Implemented protections include:

1. Stable prediction over multiple frames
2. Minimum prediction margin
3. Neutral-state requirement
4. Re-arm requirement
5. Fresh gesture requirement
6. Cooldown period
7. THREE release requirement
8. Presentation boundary checking

The THREE gesture is edge-triggered so that continuously holding the gesture does not repeatedly execute the Start/Exit command.

During live PowerPoint testing, actions were also safely rejected when the slideshow was no longer running.

---

## 12. Robustness and Failure Cases

The project deliberately documents failure cases rather than reporting only successful experiments.

Observed limitations include:

* Calibration-to-live domain shift
* Stronger distribution shift for the TWO gesture
* Dependence on live/personal calibration
* Sensitivity to hand position and orientation
* Sensitivity to lighting and background conditions
* Reduced recognition coverage when safety thresholds are overly conservative
* Limited evidence for unseen-user generalization

Mitigation strategies include:

* Live calibration
* Temporal stability
* Margin filtering
* Neutral-state unlocking
* Re-arm logic
* Fresh-gesture requirements
* THREE edge-trigger protection
* Domain analysis
* Explicit rejection of uncertain commands

---

## 13. User Interface

The system includes a real-time OpenCV dashboard displaying system state and performance information.

The dashboard provides information such as:

* Current gesture
* Stable gesture
* Prediction margin
* Slideshow state
* Safety state
* Re-arm status
* THREE trigger status
* FPS
* Hand visibility
* Command count
* PowerPoint status
* AWS verification status

This allows both the operator and evaluator to observe important internal system states during a demonstration.

---

## 14. Evaluation Evidence

The project maintains evaluation artifacts covering:

* Gesture accuracy
* System performance
* Safety
* Domain shift
* Domain adaptation
* Real-time operation
* PowerPoint integration
* Failure cases
* Robustness analysis

The final evaluation package identifies the project as a:

**FUNCTIONAL COMPETITION-READY PROTOTYPE**

The remaining focus is primarily broader validation and final competition submission packaging.

---

## 15. Reproducibility

The project uses Python and maintains pinned dependency information through `requirements.txt`.

Validated core environment:

| Dependency   | Version |
| ------------ | ------- |
| Python       | 3.14.7  |
| OpenCV       | 5.0.0   |
| MediaPipe    | 1.0.1   |
| NumPy        | 2.5.2   |
| Pandas       | 3.0.5   |
| Scikit-learn | 1.9.0   |
| PyAutoGUI    | 0.9.39  |
| PyWin32      | 312     |
| Boto3        | 1.43.89 |

Install dependencies with:

```powershell
pip install -r requirements.txt
```

The repository contains the implementation, evaluation scripts, reports, and documentation required to reproduce the major evaluation stages.

Generated datasets, trained model files, and the MediaPipe model asset are excluded from Git where appropriate and must be supplied/generated locally according to the project instructions.

AWS credentials are intentionally excluded from the repository.

---

## 16. Responsible Use

The system is designed as a presentation-assistance interface and should not be used for safety-critical control.

Recognition uncertainty can occur because of:

* Occlusion
* Poor lighting
* Background interference
* Hand orientation
* Distance from camera
* User-specific gesture variation

The safety engine therefore favors rejection over blindly executing uncertain commands.

Performance claims are reported with their experimental conditions to avoid overstating generalization.

---

## 17. Current Limitations

The most important remaining technical limitation is **generalized multi-user robustness**.

The current **93.94%** result should be interpreted as controlled live-calibration performance.

Further validation should include:

* Multiple users
* Different hand sizes
* Different camera positions
* Different backgrounds
* Different lighting conditions
* Different distances
* Longer presentation sessions

This testing is intended to determine how well the system generalizes beyond its calibrated operating conditions.

---

## 18. Competition Readiness

The current prototype has demonstrated:

* OpenCV 5 real-time vision processing
* MediaPipe 21-point hand landmark extraction
* Machine-learning-based gesture recognition experiments
* Live-domain personalization
* Temporal and safety-based command validation
* Real-time PowerPoint automation
* Presentation boundary protection
* AWS DynamoDB session analytics
* Quantitative evaluation
* Domain-shift analysis
* Failure-case documentation
* Reproducibility through pinned dependencies

The project is structured as a competition-ready prototype, with final submission work focused on:

1. Repository cleanup
2. Final documentation
3. Demo video
4. Reproducibility verification
5. Submission packaging
6. Additional robustness validation where feasible

---

## 19. Conclusion

The project demonstrates a functional real-time computer-vision system for contactless presentation control.

The implementation combines:

**OpenCV 5 + MediaPipe + Machine Learning + Live Calibration + Safety Engineering + PowerPoint Automation + AWS DynamoDB**

The system has demonstrated:

* Real-time OpenCV 5 processing
* 21-point hand landmark extraction
* 63 raw landmark features
* Personalized gesture recognition
* Controlled live recognition of **93.94%**
* Real-time operation at approximately **25.86 FPS** in a live-domain test
* PowerPoint previous/next/start/exit automation
* Multiple false-trigger prevention mechanisms
* Domain-shift analysis
* Live-domain adaptation
* AWS DynamoDB session analytics
* Explicit failure-case and limitation analysis

The project is therefore technically functional and suitable for competition demonstration, while broader multi-user and cross-environment testing remains necessary before making generalized accuracy claims.

**Overall Status: FUNCTIONAL COMPETITION-READY PROTOTYPE**
