from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta, timezone
import boto3
import os
import xgboost as xgb
import numpy as np

MODEL_PATH = "/models/xgb_model_100.pth"  # Adjust if needed
LOW_CONFIDENCE_THRESHOLD = 0.7

def run_inference_test(**context):
    print("🔁 Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    print("🔁 Connecting to MinIO...")
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )

    print("🔍 Sampling files from 'production' bucket...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=1)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="production"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            tags = s3.get_object_tagging(Bucket="production", Key=key)["TagSet"]
            tag_dict = {t['Key']: t['Value'] for t in tags}

            timestamp = tag_dict.get("timestamp")
            if not timestamp:
                continue
            ts = datetime.fromisoformat(timestamp)
            if not (start <= ts < end):
                continue

            try:
                features = np.array([
                    float(tag_dict.get("feature1", 0.0)),
                    float(tag_dict.get("feature2", 0.0)),
                    float(tag_dict.get("feature3", 0.0)),
                    float(tag_dict.get("feature4", 0.0))
                ]).reshape(1, -1)
                preds = model.predict(xgb.DMatrix(features))
                predicted_class = np.argmax(preds[0])
                confidence = float(np.max(preds[0]))

                print(f"✅ File: {key} | Predicted Class: {predicted_class} | Confidence: {confidence:.2f}")
            except Exception as e:
                print(f"⚠️ Skipped {key} due to error: {e}")

with DAG(
    dag_id="test_dag_hello_world_inference",
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    start = EmptyOperator(task_id="start")

    infer = PythonOperator(
        task_id="run_inference_test",
        python_callable=run_inference_test
    )

    start >> infer
