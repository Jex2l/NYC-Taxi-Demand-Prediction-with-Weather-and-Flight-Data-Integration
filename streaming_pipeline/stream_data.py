#!/usr/bin/env python3
import os
import time
import json
import logging
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import boto3
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StreamingPipeline:
    def __init__(self, 
                 api_url="http://localhost:8000",
                 prediction_interval=0.1,
                 data_dir="data/production",
                 results_dir="data/predictions"):
        self.api_url = api_url
        self.prediction_interval = prediction_interval
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client
        self.s3_client = boto3.client('s3',
            endpoint_url=os.getenv('S3_ENDPOINT_URL', 'https://chi.tacc.chameleoncloud.org:8080'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.bucket_name = "object-persist-project40"
        self.prefix = "nyc_taxi_split/prod"
        self.predictions_prefix = "predictions"

    def download_monthly_data(self, year, month):
        filename = f"final_features_{year}_{month:02d}.csv"
        local_path = self.data_dir / filename
        rclone_source = f"chi_tacc:{self.bucket_name}/{self.prefix}/{filename}"
        try:
            os.system(f"rclone copy {rclone_source} {self.data_dir}")
            if local_path.exists():
                logger.info(f"✅ Successfully downloaded {filename}")
                return pd.read_csv(local_path)
            else:
                logger.error(f"❌ Rclone failed: {filename} not found")
                return None
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None

    def prepare_prediction_request(self, row):
        return {
            "location_id": int(row["location_id"]),
            "year": int(row["year"]),
            "month": int(row["month"]),
            "day": int(row["day"]),
            "hour": int(row["hour"]),
            "minute": int(row["minute"]),
            "dow": int(row["dow"]),
            "dep_now": float(row["dep_now"]),
            "dep_next_30": float(row["dep_next_30"]),
            "dep_next_60": float(row["dep_next_60"]),
            "dep_next_90": float(row["dep_next_90"]),
            "dep_next_120": float(row["dep_next_120"]),
            "arr_now": float(row["arr_now"]),
            "arr_next_30": float(row["arr_next_30"]),
            "arr_next_60": float(row["arr_next_60"]),
            "arr_next_90": float(row["arr_next_90"]),
            "arr_next_120": float(row["arr_next_120"]),
            "tmpf": float(row["tmpf"]),
            "dwpf": float(row["dwpf"]),
            "relh": float(row["relh"]),
            "feel": float(row["feel"]),
            "sknt": float(row["sknt"])
        }

    def simulate_streaming(self, df, year, month):
        logger.info(f"🔁 Starting simulation with {len(df)} records")
        predictions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Simulating {year}-{month:02d}"):
            try:
                request_data = self.prepare_prediction_request(row)
                response = requests.post(f"{self.api_url}/predict", json=request_data, timeout=30)
                response.raise_for_status()
                prediction = response.json()
                actual = {
                    "pickup_count": float(row.get("pickup_count", 0)),
                    "dropoff_count": float(row.get("dropoff_count", 0))
                }
                predictions.append({
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "request": request_data,
                    "prediction": prediction,
                    "actual": actual
                })
                time.sleep(self.prediction_interval)
            except Exception as e:
                logger.warning(f"⚠️ Error at record {idx}: {e}")
                continue

        # Save locally then upload
        filename = f"predictions_{year}_{month:02d}.jsonl"
        local_file = self.results_dir / filename
        with open(local_file, "w") as f:
            for item in predictions:
                f.write(json.dumps(item) + "\n")
        logger.info(f"✅ Saved local prediction results to {local_file}")

        # Upload to object store
        try:
            s3_key = f"{self.predictions_prefix}/{filename}"
            self.s3_client.upload_file(str(local_file), self.bucket_name, s3_key)
            logger.info(f"☁️ Uploaded predictions to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logger.error(f"❌ Failed to upload predictions: {e}")

    def run(self, year=2024, month=None):
        if month is None:
            month = datetime.now().month
        logger.info(f"📅 Processing {year}-{month:02d}")
        df = self.download_monthly_data(year, month)
        if df is not None:
            self.simulate_streaming(df, year, month)
        else:
            logger.error(f"❌ No data for {year}-{month:02d}")

if __name__ == "__main__":
    pipeline = StreamingPipeline(
        api_url=os.getenv("API_URL", "http://localhost:8000"),
        prediction_interval=0.1
    )

    for month in range(1, 13):
        pipeline.run(year=2024, month=month)
