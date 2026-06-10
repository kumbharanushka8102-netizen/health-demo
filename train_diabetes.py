import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

pickle.dump(
    model,
    open("models/diabetes.pkl","wb")
)

print("Diabetes Model Saved")
print("Number of Features:", X.shape[1])