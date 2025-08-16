from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load model and dataset (CSV)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car_data = pd.read_csv("cleaned_data_car.csv")

# Get unique categories for dropdowns
companies = sorted(car_data['company'].unique())
fuel_types = sorted(car_data['fuel_type'].unique())
car_names = sorted(car_data['name'].unique())

# Create flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
        "index.html",
        companies=companies,
        fuel_types=fuel_types,
        car_names=car_names
    )

# 🔥 New API: fetch companies for selected car
@app.route("/get_companies", methods=["POST"])
def get_companies():
    car_name = request.json.get("car_name")
    related_companies = car_data[car_data["name"] == car_name]["company"].unique().tolist()
    return jsonify(related_companies)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        name = request.form.get("name")
        company = request.form.get("company")
        year = int(request.form.get("year"))
        kms_driven = int(request.form.get("kms_driven"))
        fuel_type = request.form.get("fuel_type")

        # Convert to DataFrame for model
        input_data = pd.DataFrame(
            [[name, company, year, kms_driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"]
        )

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction_text=f"💰 Estimated Car Price: ₹{prediction:,.2f}",
            companies=companies,
            fuel_types=fuel_types,
            car_names=car_names
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"⚠️ Error: {str(e)}",
            companies=companies,
            fuel_types=fuel_types,
            car_names=car_names
        )

if __name__ == "__main__":
    app.run(debug=True)
