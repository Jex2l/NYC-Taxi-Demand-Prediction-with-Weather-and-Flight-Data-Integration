from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os, json, boto3, pandas as pd
from xgboost import XGBRegressor
import joblib

# Environment variables
MINIO_URL = os.environ.get('MINIO_URL', 'http://minio:9000')
MINIO_USER = os.environ.get('MINIO_USER', 'your-access-key')
MINIO_PASSWORD = os.environ.get('MINIO_PASSWORD', 'your-secret-key')
CLEAN_BUCKET = 'production-clean'
NOISY_BUCKET = 'production-noisy'
MODEL_BUCKET = 'models'
MODEL_KEY = 'xgb_latest.pkl'

# Retraining logic
def retrain_model():
    client = boto3.client('s3', endpoint_url=MINIO_URL,
                          aws_access_key_id=MINIO_USER,
                          aws_secret_access_key=MINIO_PASSWORD)
    dfs = []

    for bucket in [CLEAN_BUCKET, NOISY_BUCKET]:
        objs = client.list_objects_v2(Bucket=bucket).get('Contents', [])
        for o in objs:
            key = o['Key']
            obj = client.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj['Body'].read())
            df = pd.DataFrame(data)
            # assume human label stored under 'true_demand'
            if 'true_demand' not in df.columns:
                continue
            dfs.append(df)

    if not dfs:
        print("No data to train on.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    X = full_df[['hour', 'day_of_week', 'temp', 'precip', 'flight_arrivals']]
    y = full_df['true_demand']

    model = XGBRegressor(max_depth=10)
    model.fit(X, y)

    # Save model and upload
    local_path = '/tmp/xgb_latest.pkl'
    joblib.dump(model, local_path)
    client.upload_file(Filename=local_path, Bucket=MODEL_BUCKET, Key=MODEL_KEY)

# DAG definition
with DAG(
    dag_id='retrain_model',
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=15)
    },
    start_date=days_ago(1),
    schedule_interval='@weekly',
    catchup=False,
    tags=['training']
) as dag:
    train = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )
