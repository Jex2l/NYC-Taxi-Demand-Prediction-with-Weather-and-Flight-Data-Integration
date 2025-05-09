#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models.signature import infer_signature
import ray
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF/XGB ensemble on monthly feature CSVs"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing final_features_YYYY_MM.csv files"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save trained model files and logs"
    )
    parser.add_argument(
        "--mlflow-uri", default=None,
        help="MLflow tracking URI (overrides environment variable)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ─── Logging ─────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ─── MLflow setup ───────────────────────────────────
    mlflow_uri = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://129.114.109.211:8000/")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("NYC_Taxi_Demand")

    # ─── Ray init ───────────────────────────────────────
    ray.init(include_dashboard=False)
    logging.info("Ray initialized successfully.")

    # ─── Load & merge data ──────────────────────────────
    logging.info(f"Discovering CSV files in {args.input_dir}...")
    csv_pattern = os.path.join(args.input_dir, "final_features_*.csv")
    files = glob.glob(csv_pattern)
    logging.info(f"Found {len(files)} files; loading...")
    if not files:
        logging.error("No CSV files found in %s", args.input_dir)
        return
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    logging.info(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} cols")

    # ─── Ensure output directory exists ────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Feature / target split ────────────────────────────
    target_cols = ["pickup_count", "dropoff_count"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    X, y = df[feature_cols], df[target_cols]

    # ─── Train/test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info(f"Train: {len(X_train)} rows; Test: {len(X_test)} rows")

    # ─── Remote training tasks ─────────────────────────────
    @ray.remote
    def train_rf(X, y):
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        )
        model.fit(X, y)
        return model

    @ray.remote
    def train_xgb(X, y):
        model = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=50,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        )
        model.fit(X, y)
        return model

    logging.info("Launching RF & XGB training in parallel via Ray…")
    rf_ref = train_rf.remote(X_train, y_train)
    xgb_ref = train_xgb.remote(X_train, y_train)
    rf_model, xgb_model = ray.get([rf_ref, xgb_ref])
    logging.info("Models trained.")

    # ─── Ensemble & metrics ───────────────────────────────
    logging.info("Ensembling via simple average…")
    pred_rf = rf_model.predict(X_test)
    pred_xgb = xgb_model.predict(X_test)
    pred_ens = (pred_rf + pred_xgb) / 2.0

    mae = mean_absolute_error(y_test, pred_ens)
    rmse = np.sqrt(mean_squared_error(y_test, pred_ens))
    r2 = r2_score(y_test, pred_ens, multioutput="uniform_average")
    logging.info(f"Metrics → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    # ─── MLflow logging ───────────────────────────────────
    with mlflow.start_run(run_name="RF_XGB_Ensemble"):
        mlflow.log_params({
            "rf_n_estimators": 100,
            "xgb_n_estimators": 50,
            "xgb_lr": 0.1
        })
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

        signature = infer_signature(X_test, pred_ens)

        rf_path = os.path.join(args.output_dir, "rf_model_compressed.pkl")
        joblib.dump(rf_model, rf_path, compress=3)
        mlflow.log_artifact(rf_path)

        xgb_path = os.path.join(args.output_dir, "xgb_model_compressed.pkl")
        joblib.dump(xgb_model, xgb_path, compress=3)
        mlflow.log_artifact(xgb_path)

    logging.info("Run logged to MLflow. All done.")

if __name__ == "__main__":
    main()
