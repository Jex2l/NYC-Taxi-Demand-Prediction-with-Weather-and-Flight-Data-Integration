from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import os, json, requests, boto3

# Environment variables
FASTAPI_URL = os.environ.get('FASTAPI_SERVER_URL', 'http://fastapi:8000')
MINIO_URL = os.environ.get('MINIO_URL', 'http://minio:9000')
MINIO_USER = os.environ.get('MINIO_USER', 'your-access-key')
MINIO_PASSWORD = os.environ.get('MINIO_PASSWORD', 'your-secret-key')
TARGET_BUCKET = 'taxi-demand-predictions'

# Python callable for Airflow task
def predict_and_store(**context):
    client = boto3.client('s3', endpoint_url=MINIO_URL,
                          aws_access_key_id=MINIO_USER,
                          aws_secret_access_key=MINIO_PASSWORD)
    results = []
    now = datetime.now(timezone.utc)
    for i in range(24):
        dt = now + timedelta(hours=i)
        features = {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'temp': 70.0,       # TODO: fetch from weather API
            'precip': 0.0,      # TODO: fetch from weather API
            'flight_arrivals': 50  # TODO: fetch real data
        }
        resp = requests.post(f"{FASTAPI_URL}/predict", json=features)
        data = resp.json()
        results.append({
            'timestamp': dt.isoformat(),
            **features,
            'predicted_demand': data.get('predicted_demand'),
            'confidence': data.get('confidence')
        })

    # Upload JSON to MinIO
    key = f"predictions/{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    client.put_object(Bucket=TARGET_BUCKET, Key=key,
                      Body=json.dumps(results).encode('utf-8'))

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='taxi_demand_predict',
    default_args=default_args,
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval='@hourly',
    catchup=False,
    tags=['prediction']
) as dag:
    task_predict = PythonOperator(
        task_id='predict_and_store',
        python_callable=predict_and_store,
        provide_context=True
    )
