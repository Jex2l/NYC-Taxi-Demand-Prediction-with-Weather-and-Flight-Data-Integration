import os
import time
import json
import logging
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def download_monthly_data(self, year, month):
        filename = f"final_features_{year}_{month:02d}.csv"
        local_path = self.data_dir / filename

        try:
            rclone_source = f"chi_tacc:object-persist-project40/nyc_taxi_split/prod/{filename}"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {filename} using Rclone from {rclone_source}")

            exit_code = os.system(f"rclone copy {rclone_source} {local_path.parent}")
            if exit_code == 0 and local_path.exists():
                logger.info(f"✅ Successfully downloaded {filename} via Rclone")
                return pd.read_csv(local_path)
            else:
                logger.error(f"❌ Rclone failed with exit code {exit_code}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
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

    def save_prediction_result(self, year, month, results):
        filename = f"predictions_{year}_{month:02d}.jsonl"
        file_path = self.results_dir / filename

        with open(file_path, "w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved predictions to {file_path}")
        self.upload_predictions_to_object_store(file_path, year, month)

    def upload_predictions_to_object_store(self, local_path, year, month):
        remote_path = f"chi_tacc:object-persist-project40/predictions"
        exit_code = os.system(f"rclone copy {local_path} {remote_path}")

        if exit_code == 0:
            logger.info(f"✅ Uploaded predictions_{year}_{month:02d}.jsonl to {remote_path}")
        else:
            logger.error(f"❌ Failed to upload predictions: Rclone exit code {exit_code}")

    def simulate_streaming(self, df, year, month):
        logger.info(f"🔁 Starting simulation for {year}-{month:02d} with {len(df)} records")
        results = []

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

                results.append({
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "request": request_data,
                    "prediction": prediction,
                    "actual": actual
                })

                time.sleep(self.prediction_interval)

            except Exception as e:
                logger.warning(f"⚠️ Error at record {idx}: {e}")
                continue

        self.save_prediction_result(year, month, results)

    def run(self, year=2024):
        for month in range(1, 13):
            df = self.download_monthly_data(year, month)
            if df is not None:
                self.simulate_streaming(df, year, month)
            else:
                logger.warning(f"Skipping month {month:02d} due to missing data")

if __name__ == "__main__":
    pipeline = StreamingPipeline(
        api_url=os.getenv('API_URL', 'http://localhost:8000'),
        prediction_interval=0.1
    )
    pipeline.run(year=2024)
