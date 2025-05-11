#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="POST a JSON payload to /predict with the features listed below",
    version="1.0.0"
)

# ─── Load the trained model ───────────────────────────────────────────────────
MODEL_PATH = "models/xgb_model_100.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ─── Feature list (exact names from your CSVs minus the targets) ──────────────
FEATURE_COLUMNS: List[str] = [
    "location_id", "year", "month", "day", "hour", "minute", "dow",
    "dep_now", "dep_next_30", "dep_next_60", "dep_next_90", "dep_next_120",
    "arr_now", "arr_next_30", "arr_next_60", "arr_next_90", "arr_next_120",
    "tmpf", "dwpf", "relh", "feel", "sknt",
]

# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    location_id: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    dow: int
    dep_now: float
    dep_next_30: float
    dep_next_60: float
    dep_next_90: float
    dep_next_120: float
    arr_now: float
    arr_next_30: float
    arr_next_60: float
    arr_next_90: float
    arr_next_120: float
    tmpf: float
    dwpf: float
    relh: float
    feel: float
    sknt: float

class PredictResponse(BaseModel):
    pickup_count: float
    dropoff_count: float

# ─── Root endpoint for info ───────────────────────────────────────────────────
@app.get("/", summary="API info")
def root():
    return {
        "message": "NYC Taxi Demand Prediction Service (FastAPI)",
        "features": FEATURE_COLUMNS,
        "endpoint": "/predict (POST)"
    }

# ─── Prediction endpoint ─────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict pickup & dropoff counts"
)
def predict(req: PredictRequest):
    # 1. Build DataFrame from the validated request
    data = req.dict()
    X = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    # 2. Run model.predict
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Unpack and return
    pickup_pred, dropoff_pred = preds[0]
    return PredictResponse(
        pickup_count=float(pickup_pred),
        dropoff_count=float(dropoff_pred)
    )

# ─── Local dev server ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
