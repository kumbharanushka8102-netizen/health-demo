import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets/kidney.csv")

# Fix ? and \t? values
df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)
df = df.replace("?", np.nan)

# Fill missing values
df = df.ffill()
df = df.bfill()

print("===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST 5 ROWS =====")
print(df.head())


# Manual Encoding

df["rbc"] = df["rbc"].map({
    "normal":1,
    "abnormal":0
})

df["pc"] = df["pc"].map({
    "normal":1,
    "abnormal":0
})

df["pcc"] = df["pcc"].map({
    "present":1,
    "notpresent":0
})

df["ba"] = df["ba"].map({
    "present":1,
    "notpresent":0
})

df["htn"] = df["htn"].map({
    "yes":1,
    "no":0
})

df["dm"] = df["dm"].map({
    "yes":1,
    "no":0
})

df["cad"] = df["cad"].map({
    "yes":1,
    "no":0
})

df["appet"] = df["appet"].map({
    "good":1,
    "poor":0
})

df["pe"] = df["pe"].map({
    "yes":1,
    "no":0
})

df["ane"] = df["ane"].map({
    "yes":1,
    "no":0
})

df["classification"] = df["classification"].map({
    "ckd":1,
    "notckd":0
})

df = df.fillna(0)

X = df.drop(["id", "classification"], axis=1)
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pickle.dump(
    model,
    open("models/kidney.pkl", "wb")
)

print("Kidney Model Saved Successfully")