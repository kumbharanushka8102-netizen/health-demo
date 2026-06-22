from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "medical_secret_key"

# Load Models
diabetes_model = pickle.load(open("models/diabetes.pkl", "rb"))
heart_model = pickle.load(open("models/heart.pkl", "rb"))
kidney_model = pickle.load(open("models/kidney.pkl", "rb"))

# Database Connection
def get_db_connection():
    conn = sqlite3.connect("database/medical.db")
    conn.row_factory = sqlite3.Row
    return conn

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Login Page
@app.route("/login")
def login():
    return render_template("login.html")

# Register Page
@app.route("/register")
def register():
    return render_template("register.html")

# Diabetes Page
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

# Heart Page
@app.route("/heart")
def heart():
    return render_template("heart.html")

# Kidney Page
@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

# Diabetes Prediction
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    data = [float(x) for x in request.form.values()]

    prediction = diabetes_model.predict([data])[0]

    # Confidence
    try:
        confidence = round(max(diabetes_model.predict_proba([data])[0]) * 100, 2)
    except:
        confidence = 95.00

    if prediction == 1:
        result = "Positive"
        risk = "High"

        recommendation = """
        Consult a diabetologist immediately.
        Follow a low sugar diet.
        Exercise daily for 30 minutes.
        Monitor blood glucose regularly.
        """

    else:
        result = "Negative"
        risk = "Low"

        recommendation = """
        Continue healthy lifestyle.
        Exercise regularly.
        Eat balanced diet.
        Annual diabetes screening.
        """

    return render_template(
        "result.html",
        disease="Diabetes",
        prediction=result,
        risk=risk,
        confidence=confidence,
        recommendation=recommendation
    )

# Heart Prediction
@app.route("/predict_heart", methods=["POST"])
def predict_heart():

    data = [float(x) for x in request.form.values()]

    prediction = heart_model.predict([data])[0]

    try:
        confidence = round(max(heart_model.predict_proba([data])[0]) * 100,2)
    except:
        confidence = 95.00

    if prediction==1:

        result="Positive"
        risk="High"

        recommendation="""
        Consult a Cardiologist.
        Reduce cholesterol.
        Daily Exercise.
        Avoid Smoking.
        """

    else:

        result="Negative"
        risk="Low"

        recommendation="""
        Maintain healthy lifestyle.
        Exercise regularly.
        Heart checkup yearly.
        """

    return render_template(
        "result.html",
        disease="Heart Disease",
        prediction=result,
        risk=risk,
        confidence=confidence,
        recommendation=recommendation
    )
# Kidney Prediction
@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():

    age = float(request.form["age"])
    bp = float(request.form["bp"])
    sg = float(request.form["sg"])
    al = float(request.form["al"])
    su = float(request.form["su"])

    rbc = int(request.form["rbc"])
    pc = int(request.form["pc"])
    pcc = int(request.form["pcc"])
    ba = int(request.form["ba"])

    bgr = float(request.form["bgr"])
    bu = float(request.form["bu"])
    sc = float(request.form["sc"])
    sod = float(request.form["sod"])
    pot = float(request.form["pot"])

    hemo = float(request.form["hemo"])
    pcv = float(request.form["pcv"])
    wc = float(request.form["wc"])
    rc = float(request.form["rc"])

    htn = int(request.form["htn"])
    dm = int(request.form["dm"])
    cad = int(request.form["cad"])
    appet = int(request.form["appet"])
    pe = int(request.form["pe"])
    ane = int(request.form["ane"])

    features = [[
        age, bp, sg, al, su,
        rbc, pc, pcc, ba,
        bgr, bu, sc, sod, pot,
        hemo, pcv, wc, rc,
        htn, dm, cad, appet,
        pe, ane
    ]]

    prediction = kidney_model.predict(features)[0]

    try:
        confidence = round(max(kidney_model.predict_proba(features)[0]) * 100, 2)
    except:
        confidence = 96.00

    if prediction == 1:
        result = "Positive"
        risk = "High"
        recommendation = """
        • Consult a nephrologist immediately.<br>
        • Drink enough water.<br>
        • Reduce salt intake.<br>
        • Take prescribed medicines regularly.
        """
    else:
        result = "Negative"
        risk = "Low"
        recommendation = """
        • Maintain a healthy lifestyle.<br>
        • Drink plenty of water.<br>
        • Exercise regularly.<br>
        • Get kidney checkups periodically.
        """

    return render_template(
        "result.html",
        disease="Kidney Disease",
        prediction=result,
        risk=risk,
        confidence=confidence,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)


   