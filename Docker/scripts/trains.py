#!/usr/bin/env python3
import argparse, glob, logging, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models.signature import infer_signature
import ray, joblib

def parse_args():
    p = argparse.ArgumentParser(description="Train RF/XGB ensemble on monthly feature CSVs")
    p.add_argument("--input-dir",  required=True, help="Directory containing feature CSVs")
    p.add_argument("--output-dir", required=True, help="Where to save model artifacts")
    p.add_argument("--mlflow-uri", default="http://129.114.109.211:8000/", 
                   help="MLflow tracking URI")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("NYC_Taxi_Demand")

    # Ray init
    ray.init(include_dashboard=False)
    logging.info("Ray initialized.")

    # Load data
    pattern = os.path.join(args.input_dir, "final_features_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.error("No CSVs in %s", args.input_dir)
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logging.info("Loaded %d×%d", *df.shape)

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Split features and targets
    target = ["pickup_count","dropoff_count"]
    X, y = df.drop(columns=target), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info("Train/Test: %d/%d rows", len(X_train), len(X_test))

    # Define remote training functions
    @ray.remote
    def train_rf(X, y):
        m = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        )
        m.fit(X, y)
        return m

    @ray.remote
    def train_xgb(X, y):
        m = MultiOutputRegressor(
            XGBRegressor(objective="reg:squarederror",
                         n_estimators=50, learning_rate=0.1,
                         random_state=42, n_jobs=-1)
        )
        m.fit(X, y)
        return m

    # Launch training
    rf_ref = train_rf.remote(X_train, y_train)
    xgb_ref = train_xgb.remote(X_train, y_train)
    rf_model, xgb_model = ray.get([rf_ref, xgb_ref])
    logging.info("Models trained.")

    # Ensemble predictions
    pred_rf = rf_model.predict(X_test)
    pred_xgb = xgb_model.predict(X_test)
    pred_ens = (pred_rf + pred_xgb) / 2

    # Compute metrics
    mae = mean_absolute_error(y_test, pred_ens)
    rmse = np.sqrt(mean_squared_error(y_test, pred_ens))
    r2   = r2_score(y_test, pred_ens, multioutput="uniform_average")
    logging.info("Metrics MAE:%.2f RMSE:%.2f R2:%.3f", mae, rmse, r2)

    # Log run to MLflow
    with mlflow.start_run(run_name="RF_XGB_Ensemble"):
        mlflow.log_params({"rf_n":100,"xgb_n":50,"xgb_lr":0.1})
        mlflow.log_metrics({"MAE":mae,"RMSE":rmse,"R2":r2})
        signature = infer_signature(X_test, pred_ens)

        for name, model in [("rf", rf_model), ("xgb", xgb_model)]:
            path = os.path.join(args.output_dir, f"{name}_model.pkl")
            joblib.dump(model, path, compress=3)
            mlflow.log_artifact(path)

if __name__ == "__main__":
    main()

