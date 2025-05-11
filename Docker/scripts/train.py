#!/usr/bin/env python3
import glob, logging
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow, mlflow.sklearn
from mlflow.models.signature import infer_signature
import ray
import joblib

# ─── 1. Logging ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ─── 2. MLflow setup ───────────────────────────────────
MLFLOW_URI = "http://129.114.109.202:8000/"
mlflow.set_tracking_uri(MLFLOW_URI)
EXPERIMENT_NAME = "NYC_Taxi_Demand"
mlflow.set_experiment(EXPERIMENT_NAME)

# ─── 3. Ray init ───────────────────────────────────────
ray.init(include_dashboard=False)
logging.info("Ray initialized successfully.")

# ─── 4. Load & merge data ──────────────────────────────
logging.info("Discovering CSV files...")
files = glob.glob("../../nyc_taxi_split/train/final_features_*.csv")
logging.info(f"Found {len(files)} files; loading...")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
logging.info(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} cols")

# ─── 5. Feature / target split ─────────────────────────
TARGET_COLS = ["pickup_count","dropoff_count"]
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]
X, y = df[FEATURE_COLS], df[TARGET_COLS]

# ─── 6. Train/test split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info(f"Train: {len(X_train)} rows; Test: {len(X_test)} rows")

# ─── 7. Ray‑accelerated training tasks ─────────────────
@ray.remote
def train_rf(X, y):
    m = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=42))
    m.fit(X, y)
    return m

@ray.remote
def train_xgb(X, y):
    m = MultiOutputRegressor(XGBRegressor(
        objective="reg:squarederror",
        n_estimators=50, learning_rate=0.1, random_state=42,
        n_jobs=-1
    ))
    m.fit(X, y)
    return m

# dispatch
logging.info("Launching RF & XGB training in parallel via Ray…")
rf_ref = train_rf.remote(X_train, y_train)
xgb_ref = train_xgb.remote(X_train, y_train)

# collect
rf_model = ray.get(rf_ref)
xgb_model = ray.get(xgb_ref)
logging.info("Models trained.")

# ─── 8. Ensemble & metrics ────────────────────────────
logging.info("Ensembling via simple average…")
pred_rf = rf_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)
pred_ens = (pred_rf + pred_xgb) / 2.0

mae = mean_absolute_error(y_test, pred_ens)
rmse = np.sqrt(mean_squared_error(y_test, pred_ens))
r2 = r2_score(y_test, pred_ens, multioutput="uniform_average")

logging.info(f"Metrics → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

# ─── 9. MLflow logging ─────────────────────────────────
with mlflow.start_run(run_name="RF_XGB_Stacking"):
    mlflow.log_params({
      "rf_n_estimators": 100,
      "xgb_n_estimators": 100,
      "xgb_lr": 0.1
    })
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

    from mlflow.models.signature import infer_signature

    signature_rf = infer_signature(X_test, pred_rf)
    signature_xgb = infer_signature(X_test, pred_xgb)

    # mlflow.sklearn.log_model(rf_model, "model_rf", signature=signature_rf, input_example=X_test.iloc[:2])
    # mlflow.sklearn.log_model(xgb_model, "model_xgb", signature=signature_xgb, input_example=X_test.iloc[:2])

    # 🆕 Save and log compressed artifact
    
    joblib.dump(rf_model, "rf_model_compressed.pkl", compress=3)
    mlflow.log_artifact("rf_model_compressed.pkl")

    logging.info("Run logged to MLflow.")

logging.info("All done.")
