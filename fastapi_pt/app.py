#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─── Load your trained ensemble model ─────────────────────────────────────────
MODEL_PATH = "/mnt/data/xgb_model_100.pth"   # adjust if needed
model = joblib.load(MODEL_PATH)

# ─── Features as per your CSV (minus the two targets) ────────────────────────
FEATURE_COLUMNS = [
    "location_id",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "dow",
    "dep_now",
    "dep_next_30",
    "dep_next_60",
    "dep_next_90",
    "dep_next_120",
    "arr_now",
    "arr_next_30",
    "arr_next_60",
    "arr_next_90",
    "arr_next_120",
    "tmpf",
    "dwpf",
    "relh",
    "feel",
    "sknt",
]

# ─── Health‐check / info endpoint ────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return (
        "NYC Taxi Demand Prediction API\n\n"
        "POST a JSON payload to /predict with these features:\n"
        f"{FEATURE_COLUMNS}\n"
    )

# ─── Prediction endpoint ─────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    # 1. Input validation
    missing = [f for f in FEATURE_COLUMNS if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # 2. Build DataFrame for prediction
    X = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    # 3. Predict: model.predict returns shape (1, 2) for pickup & dropoff
    preds = model.predict(X)
    pickup_pred, dropoff_pred = preds[0]

    # 4. Return JSON
    return jsonify({
        "pickup_count": float(pickup_pred),
        "dropoff_count": float(dropoff_pred)
    })

# ─── Run server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
