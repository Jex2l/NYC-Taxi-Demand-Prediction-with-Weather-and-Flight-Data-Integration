from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import boto3
import os
import random
import xgboost as xgb
import numpy as np

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

SAMPLE_SIZE = 5
LOW_CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = "/opt/airflow/models/xgb_model_100.pth"  # Update if needed

def ensure_buckets_exist(bucket_names):
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )
    existing_buckets = {b['Name'] for b in s3.list_buckets()['Buckets']}
    for bucket in bucket_names:
        if bucket not in existing_buckets:
            print(f"Creating bucket: {bucket}")
            s3.create_bucket(Bucket=bucket)
        else:
            print(f"Bucket already exists: {bucket}")

def init_buckets_task(**context):
    ensure_buckets_exist(['production-label-wait', 'production-noisy'])

def sample_production_images(**context):
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )

    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    if context['dag_run'].external_trigger:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=30)
    else:
        start = context['data_interval_start'].astimezone(timezone.utc)
        end = context['data_interval_end'].astimezone(timezone.utc)

    low_conf, flagged, corrected, others = [], [], [], []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket='production'):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            tags = s3.get_object_tagging(Bucket='production', Key=key)['TagSet']
            tag_dict = {t['Key']: t['Value'] for t in tags}

            timestamp = tag_dict.get("timestamp")
            if not timestamp:
                continue
            ts = datetime.fromisoformat(timestamp)
            if not (start <= ts < end):
                continue

            try:
                # Replace with real feature names/keys as needed
                features = np.array([
                    float(tag_dict.get("feature1", 0.0)),
                    float(tag_dict.get("feature2", 0.0)),
                    float(tag_dict.get("feature3", 0.0)),
                    float(tag_dict.get("feature4", 0.0))
                ]).reshape(1, -1)
                dmatrix = xgb.DMatrix(features)
                preds = model.predict(dmatrix)
                predicted_class = str(np.argmax(preds[0]))
                confidence = float(np.max(preds[0]))
            except Exception as e:
                print(f"Skipping {key} due to model/feature error: {e}")
                continue

            item = {
                "key": key,
                "confidence": confidence,
                "predicted_class": predicted_class,
                "corrected_class": tag_dict.get("corrected_class", ""),
                "flagged": tag_dict.get("flagged", "false") == "true"
            }

            if confidence < LOW_CONFIDENCE_THRESHOLD:
                low_conf.append(item)
            elif item["flagged"]:
                flagged.append(item)
            elif item["corrected_class"]:
                corrected.append(item)
            else:
                others.append(item)

    # Step 1: Get existing keys in label-wait
    existing_keys = set()
    for page in paginator.paginate(Bucket='production-label-wait'):
        for obj in page.get("Contents", []):
            existing_keys.add(obj["Key"])

    # Step 2: Filter out already processed keys
    low_conf = [i for i in low_conf if i["key"] not in existing_keys]
    flagged = [i for i in flagged if i["key"] not in existing_keys]
    corrected = [i for i in corrected if i["key"] not in existing_keys]
    others = [i for i in others if i["key"] not in existing_keys]

    selected = low_conf + flagged + corrected
    remaining = [i for i in others if i["key"] not in {x["key"] for x in selected}]
    random_others = random.sample(remaining, min(SAMPLE_SIZE, len(remaining)))
    selected += random_others

    all_items = selected + remaining
    context['ti'].xcom_push(key='selected_images', value=selected)
    context['ti'].xcom_push(key='all_images', value=all_items)

def move_sampled_images(**context):
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )

    selected = context['ti'].xcom_pull(key='selected_images', task_ids='sample_production_images')
    all_items = context['ti'].xcom_pull(key='all_images', task_ids='sample_production_images')
    selected_keys = {item['key'] for item in selected}

    for item in all_items:
        source_key = item['key']
        target_bucket = 'production-label-wait' if source_key in selected_keys else 'production-noisy'
        s3.copy_object(
            Bucket=target_bucket,
            CopySource={'Bucket': 'production', 'Key': source_key},
            Key=source_key
        )
        # Optional: delete from source
        # s3.delete_object(Bucket='production', Key=source_key)

with DAG(
    dag_id='model_inference_move_images',
    default_args=default_args,
    description='Use XGBoost model to sample and move images by confidence',
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval='@daily',
    catchup=False,
) as dag:

    init_buckets = PythonOperator(
        task_id='init_buckets',
        python_callable=init_buckets_task
    )

    sample_task = PythonOperator(
        task_id='sample_production_images',
        python_callable=sample_production_images
    )

    move_task = PythonOperator(
        task_id='move_sampled_images',
        python_callable=move_sampled_images
    )

    init_buckets >> sample_task >> move_task
