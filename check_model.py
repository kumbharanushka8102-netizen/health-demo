import pickle

model = pickle.load(open("models/diabetes.pkl", "rb"))

print("Features expected:", model.n_features_in_)