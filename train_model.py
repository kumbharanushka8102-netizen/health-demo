import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("datasets/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("models/diabetes.pkl", "wb"))