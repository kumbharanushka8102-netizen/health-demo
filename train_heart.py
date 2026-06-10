import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

pickle.dump(
    model,
    open("models/heart.pkl","wb")
)

print("Heart Model Saved")