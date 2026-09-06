import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE = Path("C:/Users/sneha/OneDrive/Desktop/AI_Hand_Gesture_Presentation")
CAL = BASE / "dataset" / "v16_calibration" / "v16_live_calibration_one_two_three.csv"
VAL = BASE / "dataset" / "v16_calibration" / "v16_balanced_live_validation.csv"

def points(row):
    return np.array([
        [float(row[f"x_{i}"]), float(row[f"y_{i}"]), float(row[f"z_{i}"])]
        for i in range(21)
    ])

def finger_directions(row):
    p = points(row)
    p -= p[0]

    wrist = p[0]
    forward = p[9] - wrist
    forward /= max(np.linalg.norm(forward),1e-8)

    side = p[17] - p[5]
    side /= max(np.linalg.norm(side),1e-8)

    normal = np.cross(side,forward)
    normal /= max(np.linalg.norm(normal),1e-8)

    side = np.cross(forward,normal)
    side /= max(np.linalg.norm(side),1e-8)

    chains = {
        "thumb":[1,2,3,4],
        "index":[5,6,7,8],
        "middle":[9,10,11,12],
        "ring":[13,14,15,16],
        "pinky":[17,18,19,20]
    }

    result={}

    for name,chain in chains.items():
        vals=[]

        for a,b in zip(chain[:-1],chain[1:]):
            v=p[b]-p[a]
            v/=max(np.linalg.norm(v),1e-8)

            vals.extend([
                np.dot(v,side),
                np.dot(v,forward),
                np.dot(v,normal)
            ])

        result[name]=np.array(vals)

    return result

cal=pd.read_csv(CAL)
val=pd.read_csv(VAL)

FC=[finger_directions(r) for _,r in cal.iterrows()]
FV=[finger_directions(r) for _,r in val.iterrows()]

yc=cal["label"].values
yv=val["label"].values

print("="*78)
print("V16 STEP 16 - ONE/TWO/THREE SEPARATION")
print("="*78)

groups={
    "index":["index"],
    "middle":["middle"],
    "ring":["ring"],
    "pinky":["pinky"],
    "index+middle":["index","middle"],
    "index+ring":["index","ring"],
    "index+pinky":["index","pinky"],
    "middle+ring":["middle","ring"],
    "middle+pinky":["middle","pinky"],
    "ring+pinky":["ring","pinky"],
    "index+middle+ring":["index","middle","ring"],
    "index+middle+pinky":["index","middle","pinky"],
    "index+ring+pinky":["index","ring","pinky"],
    "middle+ring+pinky":["middle","ring","pinky"],
    "index+middle+ring+pinky":["index","middle","ring","pinky"]
}

results=[]

for name,selected in groups.items():

    Xc=np.vstack([
        np.concatenate([d[f] for f in selected])
        for d in FC
    ])

    Xv=np.vstack([
        np.concatenate([d[f] for f in selected])
        for d in FV
    ])

    for k in [1,3,5,7,9,11,15]:

        model=make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=k,
                weights="distance"
            )
        )

        model.fit(Xc,yc)
        pred=model.predict(Xv)

        acc=accuracy_score(yv,pred)

        results.append((acc,name,k,len(selected)*9))

print("\nTOP CONFIGURATIONS:")
print("-"*78)

for acc,name,k,nf in sorted(results,reverse=True)[:20]:
    print(
        f"{acc*100:6.2f}% | "
        f"{name:32s} | "
        f"k={k:2d} | "
        f"features={nf}"
    )

best=max(results)

print("\n"+"="*78)
print("BEST")
print("="*78)

print(f"\nFeature family : {best[1]}")
print(f"KNN k          : {best[2]}")
print(f"Transfer       : {best[0]*100:.2f}%")

selected=groups[best[1]]

Xc=np.vstack([
    np.concatenate([d[f] for f in selected])
    for d in FC
])

Xv=np.vstack([
    np.concatenate([d[f] for f in selected])
    for d in FV
])

model=make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=best[2],
        weights="distance"
    )
)

model.fit(Xc,yc)
pred=model.predict(Xv)

print("\nConfusion matrix:")
print(confusion_matrix(
    yv,pred,
    labels=["one","two","three"]
))

print("\nPer-gesture:")

for g in ["one","two","three"]:
    mask=yv==g
    correct=np.sum(pred[mask]==g)

    print(
        f"{g.upper():6s}: "
        f"{correct/np.sum(mask)*100:6.2f}% "
        f"({correct}/{np.sum(mask)})"
    )

print("\n"+"="*78)
print("ONE-SPECIFIC ANALYSIS")
print("="*78)

# Binary ONE vs non-ONE using best representation
binary_yc=np.where(yc=="one","one","other")
binary_yv=np.where(yv=="one","one","other")

binary_model=make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=best[2],
        weights="distance"
    )
)

binary_model.fit(Xc,binary_yc)
binary_pred=binary_model.predict(Xv)

binary_acc=accuracy_score(binary_yv,binary_pred)

print(f"\nONE vs NON-ONE accuracy: {binary_acc*100:.2f}%")

print("\nONE detection:")
one_mask=binary_yv=="one"

print(
    f"ONE correctly detected: "
    f"{np.sum(binary_pred[one_mask]=='one')}/"
    f"{np.sum(one_mask)}"
)

print(
    f"ONE missed: "
    f"{np.sum(binary_pred[one_mask]!='one')}/"
    f"{np.sum(one_mask)}"
)

print("\nNon-ONE falsely detected as ONE:")

other_mask=binary_yv=="other"

print(
    f"{np.sum(binary_pred[other_mask]=='one')}/"
    f"{np.sum(other_mask)}"
)

print("\n"+"="*78)
print("DECISION")
print("="*78)

if best[0]>=0.75:
    print("\nVERY PROMISING")
    print("Use this representation for a hierarchical recognizer.")

elif best[0]>=0.65:
    print("\nPROMISING")
    print("ONE needs specialized handling before automation.")

else:
    print("\nWEAK")
    print("Further representation work is required.")

print("\nNo new data collected.")
print("No existing data modified.")
print("="*78)

input("\nPress Enter to return to PowerShell...")
