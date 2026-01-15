from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# ---------------- LOAD MODEL & PREPROCESSORS ---------------- #

with open("rain_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("cat_imputer.pkl", "rb") as f:
    cat_imputer = pickle.load(f)

with open("columns.pkl", "rb") as f:
    cols = pickle.load(f)

numeric_cols = cols["numeric_cols"]
cato_cols = cols["cato_cols"]
encoded_cols = cols["encoded_cols"]

# ---------------- ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        user_input = {}

        # -------- NUMERIC INPUTS (empty -> NaN) -------- #
        for col in numeric_cols:
            value = request.form.get(col)
            user_input[col] = float(value) if value not in ["", None] else np.nan

        # -------- CATEGORICAL INPUTS (empty -> NaN) -------- #
        for col in cato_cols:
            value = request.form.get(col)
            user_input[col] = value if value not in ["", None] else np.nan

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # -------- PREPROCESSING -------- #
        input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        input_df[cato_cols] = cat_imputer.transform(input_df[cato_cols])
        input_df[encoded_cols] = encoder.transform(input_df[cato_cols])

        X_input = input_df[numeric_cols + encoded_cols]

        # -------- PREDICTION -------- #
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]

        prediction = pred
        probability = round(prob * 100, 2)

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)
