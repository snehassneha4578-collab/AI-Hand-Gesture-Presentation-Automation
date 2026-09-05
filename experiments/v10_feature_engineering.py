import pandas as pd
import numpy as np
import os

BASE_DIR = r"C:\Users\sneha\OneDrive\Desktop\AI_Hand_Gesture_Presentation"
INPUT = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v9.csv")
OUTPUT = os.path.join(BASE_DIR, "dataset", "gesture_dataset_v10_features.csv")

df = pd.read_csv(INPUT)

labels = df["label"].values
sessions = df["session_id"].values

feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]

X_raw = df[feature_cols].values.reshape(-1, 21, 3)

# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

INDEX_DIP = 7
MIDDLE_DIP = 11
RING_DIP = 15
PINKY_DIP = 19

THUMB_IP = 3

def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom == 0:
        return 0.0

    value = np.dot(ba, bc) / denom
    value = np.clip(value, -1.0, 1.0)

    return np.arccos(value)

def create_features(hand):

    wrist = hand[WRIST]

    # Palm scale
    palm_scale = distance(
        wrist,
        hand[MIDDLE_MCP]
    )

    if palm_scale < 1e-6:
        palm_scale = 1.0

    # Wrist-centered + scale normalized coordinates
    normalized = (hand - wrist) / palm_scale

    features = []

    # ---------------------------------------------------------
    # NORMALIZED LANDMARK GEOMETRY
    # ---------------------------------------------------------

    features.extend(normalized.flatten())

    # ---------------------------------------------------------
    # FINGER TIP DISTANCES FROM WRIST
    # ---------------------------------------------------------

    tips = [
        THUMB_TIP,
        INDEX_TIP,
        MIDDLE_TIP,
        RING_TIP,
        PINKY_TIP
    ]

    for tip in tips:
        features.append(
            distance(hand[tip], wrist) / palm_scale
        )

    # ---------------------------------------------------------
    # FINGER SEGMENT LENGTHS
    # ---------------------------------------------------------

    finger_chains = [
        [THUMB_TIP, THUMB_IP, 2, 1],
        [INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP],
        [MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP],
        [RING_TIP, RING_DIP, RING_PIP, RING_MCP],
        [PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP]
    ]

    for chain in finger_chains:

        for i in range(len(chain) - 1):

            d = distance(
                hand[chain[i]],
                hand[chain[i + 1]]
            )

            features.append(d / palm_scale)

    # ---------------------------------------------------------
    # FINGER EXTENSION RATIOS
    # ---------------------------------------------------------

    finger_data = [
        (INDEX_TIP, INDEX_PIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP, RING_PIP, RING_MCP),
        (PINKY_TIP, PINKY_PIP, PINKY_MCP)
    ]

    for tip, pip, mcp in finger_data:

        tip_wrist = distance(hand[tip], wrist)
        pip_wrist = distance(hand[pip], wrist)
        mcp_wrist = distance(hand[mcp], wrist)

        features.append(
            tip_wrist / max(mcp_wrist, 1e-6)
        )

        features.append(
            tip_wrist / max(pip_wrist, 1e-6)
        )

    # Thumb extension
    features.append(
        distance(hand[THUMB_TIP], wrist) /
        max(distance(hand[THUMB_IP], wrist), 1e-6)
    )

    # ---------------------------------------------------------
    # JOINT ANGLES
    # ---------------------------------------------------------

    angle_triplets = [
        (2, 3, 4),
        (5, 6, 7),
        (6, 7, 8),
        (9, 10, 11),
        (10, 11, 12),
        (13, 14, 15),
        (14, 15, 16),
        (17, 18, 19),
        (18, 19, 20)
    ]

    for a, b, c in angle_triplets:
        features.append(
            angle(hand[a], hand[b], hand[c])
        )

    # ---------------------------------------------------------
    # TIP-TO-TIP DISTANCES
    # ---------------------------------------------------------

    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):

            features.append(
                distance(
                    hand[tips[i]],
                    hand[tips[j]]
                ) / palm_scale
            )

    # ---------------------------------------------------------
    # TIP HEIGHT RELATIONSHIPS
    # ---------------------------------------------------------

    for i in range(1, len(tips)):

        features.append(
            (hand[tips[i]][1] - hand[MIDDLE_MCP][1])
            / palm_scale
        )

    # ---------------------------------------------------------
    # TIP X RELATIONSHIPS
    # ---------------------------------------------------------

    for i in range(1, len(tips)):

        features.append(
            (hand[tips[i]][0] - hand[INDEX_MCP][0])
            / palm_scale
        )

    # ---------------------------------------------------------
    # FINGER CURL INDICATORS
    # ---------------------------------------------------------

    for tip, pip, mcp in finger_data:

        d_tip = distance(hand[tip], hand[mcp])
        d_pip = distance(hand[pip], hand[mcp])

        features.append(
            d_tip / max(d_pip, 1e-6)
        )

    # ---------------------------------------------------------
    # THUMB-INDEX PINCH FEATURES
    # ---------------------------------------------------------

    thumb_index = distance(
        hand[THUMB_TIP],
        hand[INDEX_TIP]
    ) / palm_scale

    thumb_middle = distance(
        hand[THUMB_TIP],
        hand[MIDDLE_TIP]
    ) / palm_scale

    features.append(thumb_index)
    features.append(thumb_middle)

    # Binary-style geometric states
    features.append(float(thumb_index < 0.35))
    features.append(float(thumb_middle < 0.45))

    # ---------------------------------------------------------
    # PALM WIDTH / HEIGHT
    # ---------------------------------------------------------

    palm_width = distance(
        hand[INDEX_MCP],
        hand[PINKY_MCP]
    ) / palm_scale

    palm_height = distance(
        hand[WRIST],
        hand[MIDDLE_MCP]
    ) / palm_scale

    features.append(palm_width)
    features.append(palm_height)

    return features

print("=" * 75)
print("V10 HYBRID FEATURE ENGINEERING")
print("=" * 75)

all_features = []

for hand in X_raw:
    all_features.append(create_features(hand))

X = np.asarray(all_features, dtype=np.float64)

print(f"\nInput rows: {len(df)}")
print(f"Input features: {len(feature_cols)}")
print(f"Output features: {X.shape[1]}")
print(f"Sessions: {df['session_id'].nunique()}")

valid = np.isfinite(X).all(axis=1)

print(f"Valid rows: {valid.sum()}/{len(valid)}")

if not valid.all():
    X = X[valid]
    labels = labels[valid]
    sessions = sessions[valid]

out = pd.DataFrame(
    X,
    columns=[f"f_{i}" for i in range(X.shape[1])]
)

out["label"] = labels
out["session_id"] = sessions

out.to_csv(OUTPUT, index=False)

print(f"\nSaved to:")
print(OUTPUT)

print("\n" + "=" * 75)
print("V10 FEATURE ENGINEERING COMPLETE")
print("=" * 75)
