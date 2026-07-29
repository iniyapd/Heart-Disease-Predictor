from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("heart_disease_knn_pipeline.pkl")
@app.route("/")
def home():
    return render_template("index.html", prediction=None, description=None, color=None, confidence=None)
@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form
    data = {
    "age": float(form_data["age"]),
    "sex": form_data["sex"],
    "cp": form_data["cp"],
    "trestbps": float(form_data["trestbps"]),
    "chol": float(form_data["chol"]),
    "fbs": form_data["fbs"]=="True",
    "restecg": form_data["restecg"],
    "thalch": float(form_data["thalch"]),
    "exang": form_data["exang"]=="True",
    "oldpeak": float(form_data["oldpeak"]),
    "slope": form_data["slope"],
    "ca": float(form_data["ca"]),
    "thal": form_data["thal"]
}
    if not (18 <= data["age"] <= 120):
        return render_template(
            "index.html",
            prediction="Invalid Age!!",
            description="Age must be between 18 and 120 years.",
            color="red",
            confidence=None
        )

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities) * 100

    if prediction==0:
        result = "No Heart disease detected."
        description = "The model predicts a low likelihood of heart disease."
        color = "green"

    elif prediction == 1:
        result = "Mild Heart Disease"
        description = "The model predicts signs consistent with mild heart disease."
        color = "yellow"

    elif prediction == 2:
        result = "Moderate Heart Disease"
        description = "The model predicts signs consistent with moderate heart disease."
        color = "orange"

    elif prediction == 3:
        result = "Severe Heart Disease"
        description = "The model predicts signs consistent with severe heart disease."
        color = "red"

    else:
        result = "Very Severe Heart Disease"
        description = "The model predicts signs consistent with very severe heart disease."
        color = "darkred"

    return render_template("index.html", prediction=result, description=description, color=color, confidence=round(confidence, 2))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)