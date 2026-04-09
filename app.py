from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("data/churn_model_clean.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age = data["age"]
        account_manager = data["account_manager"]
        years = data["years"]
        num_sites = data["num_sites"]

        # # Vérification de la cohérence des données
        # if age - years < 18:
        #     return jsonify({
        #         "error": "Invalid input: age must be at least 18 years greater than years of service."
        #     }), 400  # HTTP 400 Bad Request

        # Préparer les features pour le modèle
        features = np.array([[age, account_manager, years, num_sites]])

        # Prédiction
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(proba)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # HTTP 500 Internal Server Error


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)