from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os, json, random, boto3, requests

# Environment variables
MINIO_URL = os.environ.get('MINIO_URL', 'http://minio:9000')
MINIO_USER = os.environ.get('MINIO_USER', 'your-access-key')
MINIO_PASSWORD = os.environ.get('MINIO_PASSWORD', 'your-secret-key')
SOURCE_BUCKET = 'taxi-demand-predictions'
LABEL_WAIT_BUCKET = 'production-label-wait'
NOISY_BUCKET = 'production-noisy'

# Label Studio config\LS_URL = os.environ.get('LS_URL', 'http://label-studio:8080')
LS_TOKEN = os.environ.get('LS_TOKEN', 'your-label-studio-token')
PROJECT_ID = os.environ.get('LS_PROJECT_ID', '1')

# Initialize buckets if missing
def init_buckets():
    client = boto3.client('s3', endpoint_url=MINIO_URL,
                          aws_access_key_id=MINIO_USER,
                          aws_secret_access_key=MINIO_PASSWORD)
    for bucket in [LABEL_WAIT_BUCKET, NOISY_BUCKET]:
        try:
            client.head_bucket(Bucket=bucket)
        except client.exceptions.ClientError:
            client.create_bucket(Bucket=bucket)

# Sample low-confidence + random predictions

def sample_for_labeling(**context):
    client = boto3.client('s3', endpoint_url=MINIO_URL,
                          aws_access_key_id=MINIO_USER,
                          aws_secret_access_key=MINIO_PASSWORD)
    objs = client.list_objects_v2(Bucket=SOURCE_BUCKET, Prefix='predictions/').get('Contents', [])
    keys = [o['Key'] for o in objs]

    # Low-confidence threshold
    low_conf = []
    for key in keys:
        obj = client.get_object(Bucket=SOURCE_BUCKET, Key=key)
        data = json.loads(obj['Body'].read())
        avg_conf = sum(item['confidence'] for item in data) / len(data)
        if avg_conf < 0.7:
            low_conf.append(key)

    # Random sample of 10 files
    rnd_sample = random.sample(keys, min(len(keys), 10))
    to_label = list(set(low_conf + rnd_sample))

    # Move selected to label_wait bucket
    for key in to_label:
        copy_src = {'Bucket': SOURCE_BUCKET, 'Key': key}
        client.copy_object(Bucket=LABEL_WAIT_BUCKET, CopySource=copy_src, Key=key)

    # Create Label Studio tasks
    tasks = [{'data': {'json_url': f"{MINIO_URL}/{LABEL_WAIT_BUCKET}/{key}"}} for key in to_label]
    headers = {'Authorization': f'Token {LS_TOKEN}'}
    import_url = f"{LS_URL}/api/projects/{PROJECT_ID}/import"
    requests.post(import_url, headers=headers, json=tasks)

# DAG definition
with DAG(
    dag_id='sample_for_labeling',
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    start_date=days_ago(1),
    schedule_interval='@hourly',
    catchup=False,
    tags=['labeling']
) as dag:
    init = PythonOperator(
        task_id='init_buckets',
        python_callable=init_buckets
    )
    sample = PythonOperator(
        task_id='sample_for_labeling',
        python_callable=sample_for_labeling,
        provide_context=True
    )

    init >> sample
