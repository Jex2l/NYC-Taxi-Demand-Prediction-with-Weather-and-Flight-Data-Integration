import os
import json
import boto3
import pandas as pd
from pathlib import Path

def load_predictions_from_local(pred_dir):
    all_data = []
    for file in Path(pred_dir).glob("monthly_*.jsonl"):
        with open(file) as f:
            all_data.extend([json.loads(line) for line in f])
    return pd.DataFrame(all_data)

def load_predictions_from_s3(bucket, prefix, endpoint, access_key, secret_key):
    s3 = boto3.client('s3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1"
    )
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    all_data = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "monthly_" in key and key.endswith(".jsonl"):
                content = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
                lines = content.strip().split("\n")
                all_data.extend([json.loads(line) for line in lines])
    return pd.DataFrame(all_data)
