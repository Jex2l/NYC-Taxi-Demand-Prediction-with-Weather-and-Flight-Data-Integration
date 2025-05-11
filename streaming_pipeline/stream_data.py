#!/usr/bin/env python3
import os
import time
import json
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

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
                 api_url="http://localhost:8000",  # Update this to your FastAPI service URL
                 prediction_interval=900,  # 15 minutes in seconds
                 data_dir="data/production",
                 results_dir="data/predictions"):
        self.api_url = api_url
        self.prediction_interval = prediction_interval
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client for object store access
        self.s3_client = boto3.client('s3',
            endpoint_url=os.getenv('S3_ENDPOINT_URL', 'https://chi.tacc.chameleoncloud.org:8080'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.bucket_name = "object-persist-project40"
        self.prefix = "nyc_taxi_split/prod"

    def download_monthly_data(self, year, month):
        """Download monthly data from object store"""
        filename = f"final_features_{year}_{month:02d}.csv"
        local_path = self.data_dir / filename
        
        try:
            # Download from object store
            s3_key = f"{self.prefix}/{filename}"
            logger.info(f"Downloading {filename} from {self.bucket_name}/{s3_key}")
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Successfully downloaded {filename}")
            return pd.read_csv(local_path)
        except ClientError as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return None

    def prepare_prediction_request(self, row):
        """Convert DataFrame row to API request format"""
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

    def save_prediction_result(self, request, prediction, actual):
        """Save prediction results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "timestamp": timestamp,
            "request": request,
            "prediction": prediction,
            "actual": actual
        }
        
        # Save to daily file
        date_str = datetime.now().strftime("%Y%m%d")
        result_file = self.results_dir / f"predictions_{date_str}.jsonl"
        
        with open(result_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        logger.info(f"Saved prediction result to {result_file}")

    def simulate_streaming(self, df):
        """Simulate real-time data streaming"""
        logger.info(f"Starting simulation with {len(df)} records")
        
        for idx, row in df.iterrows():
            try:
                # Prepare request
                request_data = self.prepare_prediction_request(row)
                
                # Send prediction request
                logger.info(f"Sending prediction request for record {idx + 1}")
                response = requests.post(
                    f"{self.api_url}/predict",
                    json=request_data,
                    timeout=30  # Add timeout for API requests
                )
                response.raise_for_status()
                
                # Get prediction
                prediction = response.json()
                
                # Save results
                actual = {
                    "pickup_count": float(row.get("pickup_count", 0)),
                    "dropoff_count": float(row.get("dropoff_count", 0))
                }
                self.save_prediction_result(request_data, prediction, actual)
                
                logger.info(f"Processed record {idx + 1}/{len(df)}")
                
                # Simulate real-time delay (15 minutes)
                logger.info(f"Waiting {self.prediction_interval} seconds before next prediction")
                time.sleep(self.prediction_interval)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request error for record {idx}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                continue

    def run(self, year=2024, month=None):
        """Run the streaming pipeline for a specific month"""
        if month is None:
            month = datetime.now().month
            
        logger.info(f"Starting streaming pipeline for {year}-{month:02d}")
        
        # Download and process monthly data
        df = self.download_monthly_data(year, month)
        if df is not None:
            self.simulate_streaming(df)
        else:
            logger.error(f"Failed to process data for {year}-{month:02d}")

if __name__ == "__main__":
    # Example usage
    pipeline = StreamingPipeline(
        api_url=os.getenv('API_URL', 'http://localhost:8000'),
        prediction_interval=900  # 15 minutes
    )
    pipeline.run(year=2024, month=1)  # Process January 2024 data 