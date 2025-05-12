from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os, json, boto3, requests

# Environment variables
MINIO_URL = os.environ.get('MINIO_URL', 'http://minio:9000')
MINIO_USER = os.environ.get('MINIO_USER', 'your-access-key')
MINIO_PASSWORD = os.environ.get('MINIO_PASSWORD', 'your-secret-key')
LABEL_WAIT_BUCKET = 'production-label-wait'
CLEAN_BUCKET = 'production-clean'
NOISY_BUCKET = 'production-noisy'

LS_URL = os.environ.get('LS_URL', 'http://label-studio:8080')
LS_TOKEN = os.environ.get('LS_TOKEN', 'your-label-studio-token')
PROJECT_ID = os.environ.get('LS_PROJECT_ID', '1')

# Fetch completed labels and evaluate

def fetch_and_evaluate():
    # Label Studio API: get completed annotations
    headers = {'Authorization': f'Token {LS_TOKEN}'}
    tasks_url = f"{LS_URL}/api/projects/{PROJECT_ID}/tasks?completed=true"
    resp = requests.get(tasks_url, headers=headers).json()
    annotations = resp.get('data', [])

    client = boto3.client('s3', endpoint_url=MINIO_URL,
                          aws_access_key_id=MINIO_USER,
                          aws_secret_access_key=MINIO_PASSWORD)
    correct = 0
    total = 0

    for task in annotations:
        data = task['data']
        key = data['json_url'].split(f"/{LABEL_WAIT_BUCKET}/")[1]
        # load original prediction
        obj = client.get_object(Bucket=LABEL_WAIT_BUCKET, Key=key)
        pred = json.loads(obj['Body'].read())
        pred_value = pred[0]['predicted_demand']  # first entry

        # human label: assume annotation value in JSON
        human_val = None
        for ann in task.get('annotations', []):
            # Example: annotation result name 'number'
            for r in ann['result']:
                if 'number' in r.get('value', {}):
                    human_val = r['value']['number']
        if human_val is None:
            continue

        total += 1
        if abs(pred_value - human_val) <= 5:  # within tolerance
            dest = CLEAN_BUCKET
            correct += 1
        else:
            dest = NOISY_BUCKET

        # move object
        client.copy_object(Bucket=dest, CopySource={'Bucket': LABEL_WAIT_BUCKET, 'Key': key}, Key=key)
        client.delete_object(Bucket=LABEL_WAIT_BUCKET, Key=key)

    accuracy = correct / total if total else 0
    print(f"Batch accuracy: {accuracy:.2%}")

# DAG definition
with DAG(
    dag_id='fetch_labeled_and_evaluate',
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    start_date=days_ago(1),
    schedule_interval='@hourly',
    catchup=False,
    tags=['evaluation']
) as dag:
    evaluate = PythonOperator(
        task_id='fetch_and_evaluate',
        python_callable=fetch_and_evaluate
    )
