#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="POST JSON to /predict with the features below",
    version="1.0.0"
)

# ─── Load the model ───────────────────────────────────────────────────────────
MODEL_PATH = "/models/xgb_model_100.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ─── Feature list (exact names from your CSVs minus the two targets) ───────────
FEATURE_COLUMNS: List[str] = [
    "location_id", "year", "month", "day", "hour", "minute", "dow",
    "dep_now", "dep_next_30", "dep_next_60", "dep_next_90", "dep_next_120",
    "arr_now", "arr_next_30", "arr_next_60", "arr_next_90", "arr_next_120",
    "tmpf", "dwpf", "relh", "feel", "sknt",
]

# ─── Request/response schemas ─────────────────────────────────────────────────
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

# ─── Info endpoint ────────────────────────────────────────────────────────────
@app.get("/", summary="API info")
def root():
    return {
        "message": "NYC Taxi Demand Prediction (FastAPI)",
        "features": FEATURE_COLUMNS,
        "endpoint": "/predict (POST)"
    }

# ─── Prediction endpoint ─────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, summary="Predict counts")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()], columns=FEATURE_COLUMNS)
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    pickup_pred, dropoff_pred = preds[0]
    return PredictResponse(
        pickup_count=float(pickup_pred),
        dropoff_count=float(dropoff_pred)
    )

# ─── Local debug server ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
