import os
import sys
import requests
import pandas as pd
import zipfile
import glob
import time
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ========== CONFIG ========== #
if len(sys.argv) != 3:
    print("Usage: python download_data.py <year> <month>")
    sys.exit(1)

year = int(sys.argv[1])
month = int(sys.argv[2])
ym = f"{year}-{month:02d}"
ym_us = f"{year}_{month:02d}"

# Base URLs
TAXI_YELLOW_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
TAXI_GREEN_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{ym}.parquet"
BTS_FLIGHT_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

# ========== PATHS ========== #
data_root = "/app/data"
taxi_dir = os.path.join(data_root, "taxi")
flight_dir = os.path.join(data_root, "flight")
weather_dir = os.path.join(data_root, "weather")

os.makedirs(taxi_dir, exist_ok=True)
os.makedirs(flight_dir, exist_ok=True)
os.makedirs(weather_dir, exist_ok=True)

# ========== 1. TAXI DATA ========== #
output_file = os.path.join(taxi_dir, f"all_taxi_{year}_{month:02d}.csv")
if os.path.exists(output_file):
    os.remove(output_file)

for idx, taxi_type in enumerate(["yellow", "green"]):
    if taxi_type == "green" and (year < 2013 or (year == 2013 and month < 8)):
        continue

    url = TAXI_YELLOW_URL.format(ym=ym) if taxi_type == "yellow" else TAXI_GREEN_URL.format(ym=ym)
    fname = os.path.join(taxi_dir, f"{taxi_type}_{ym}.parquet")

    if not os.path.exists(fname):
        try:
            r = requests.get(url, timeout=30)
            with open(fname, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded {fname}")
        except Exception as e:
            print(f"Failed download {fname}: {e}")
            continue

    try:
        df = pd.read_parquet(fname)
        df.columns = [col.lower() for col in df.columns]

        if taxi_type == "yellow":
            if 'tpep_pickup_datetime' in df.columns:
                df = df.rename(columns={
                    'tpep_pickup_datetime': 'pickup_datetime',
                    'tpep_dropoff_datetime': 'dropoff_datetime',
                    'pulocationid': 'pickup_location_id',
                    'dolocationid': 'dropoff_location_id'
                })
            elif 'trip_pickup_datetime' in df.columns:
                df = df.rename(columns={
                    'trip_pickup_datetime': 'pickup_datetime',
                    'trip_dropoff_datetime': 'dropoff_datetime',
                    'pulocationid': 'pickup_location_id',
                    'dolocationid': 'dropoff_location_id'
                })
        else:
            if 'lpep_pickup_datetime' in df.columns:
                df = df.rename(columns={
                    'lpep_pickup_datetime': 'pickup_datetime',
                    'lpep_dropoff_datetime': 'dropoff_datetime',
                    'pulocationid': 'pickup_location_id',
                    'dolocationid': 'dropoff_location_id'
                })

        df = df[['pickup_datetime', 'dropoff_datetime', 'pickup_location_id', 'dropoff_location_id']].dropna()

        mode = 'w' if idx == 0 else 'a'
        df.to_csv(output_file, mode=mode, header=(mode == 'w'), index=False)
        print(f"Processed {fname}")

    except Exception as e:
        print(f"Failed to process {fname}: {e}")

for f in os.listdir(taxi_dir):
    if f.endswith(".parquet"):
        os.remove(os.path.join(taxi_dir, f))
print("🧹 Deleted individual taxi parquet files")
print(f"✅ Created: {output_file}")

# ========== 2. FLIGHT DATA ========== #
# ========== 2. FLIGHT DATA ========== #
output_file = os.path.join(flight_dir, f"all_flights_{year}_{month:02d}.csv")
header_written = False
zip_url = BTS_FLIGHT_URL.format(year=year, month=month)
zip_path = os.path.join(flight_dir, f"flight_{ym_us}.zip")

session = requests.Session()
retries = Retry(total=3, backoff_factor=10, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

if not os.path.exists(zip_path):
    try:
        r = session.get(zip_url, timeout=(10, 120))
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded {zip_path}")
    except Exception as e:
        print(f"Failed flight zip {ym_us}: {e}")
for f in os.listdir(flight_dir):
    if f.endswith(".csv") and f != f"all_flights_{year}_{month:02d}.csv":
        os.remove(os.path.join(flight_dir, f))
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(flight_dir)
    print(f"Extracted {zip_path}")
except Exception as e:
    print(f"Failed extracting {zip_path}: {e}")
time.sleep(3)
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extracted_files = zip_ref.namelist()
except Exception as e:
    print(f"Failed to get extracted file names: {e}")
    extracted_files = []
for extracted_name in extracted_files:
    extracted_file = os.path.join(flight_dir, extracted_name)
    try:
        df = pd.read_csv(extracted_file, low_memory=False, on_bad_lines='skip')
        if df.empty:
            print(f"⚠️  Skipped empty or invalid flight file: {extracted_file}")
            continue
        df.to_csv(output_file, mode='a', header=not header_written, index=False)
        header_written = True
    except Exception as e:
        print(f"Failed to read {extracted_file}: {e}")
for f in os.listdir(flight_dir):
    if f.endswith(".zip") or (f.endswith(".csv") and f != f"all_flights_{year}_{month:02d}.csv"):
        os.remove(os.path.join(flight_dir, f))
print("🧹 Deleted individual flight files")
print(f"✅ Created: {output_file}")

# ========== 3. WEATHER DATA ========== #
asos_configs = [
    {"stations": ["JFK", "LGA"], "network": "NY_ASOS"},
    {"stations": ["EWR"], "network": "NJ_ASOS"}
]

ASOS_URL_TEMPLATE = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    "station={station}&network={network}"
    "&year1={start_year}&month1={start_month}&day1=1"
    "&year2={end_year}&month2={end_month}&day2={end_day}"
    "&tz=Etc/UTC&format=comma&direct=no"
)

weather_files = []
output_file = os.path.join(weather_dir, f"all_weather_{year}_{month:02d}.csv")
header_written = False

startts = datetime(year, month, 1)
endts = startts + relativedelta(months=1)

for config in asos_configs:
    for station in config["stations"]:
        url = ASOS_URL_TEMPLATE.format(
            station=station,
            network=config["network"],
            start_year=startts.year,
            start_month=startts.month,
            end_year=endts.year,
            end_month=endts.month,
            end_day=endts.day
        )

        fname = f"asos_{station}_{startts.year}_{startts.month:02d}.csv"
        fpath = os.path.join(weather_dir, fname)

        if not os.path.exists(fpath):
            try:
                r = requests.get(url, timeout=60)
                if "ERROR" in r.text or r.status_code != 200:
                    print(f"Error downloading {station}: {r.status_code}")
                    continue
                with open(fpath, "wb") as f:
                    f.write(r.content)
                print(f"Saved {fpath}")
                weather_files.append(fpath)
            except Exception as e:
                print(f"Failed {station} {year}-{month:02d}: {e}")
        else:
            weather_files.append(fpath)

for f in weather_files:
    try:
        df = pd.read_csv(f, skiprows=5)
        df.to_csv(output_file, mode='a', header=not header_written, index=False)
        header_written = True
    except Exception as e:
        print(f"Could not read {f}: {e}")

for f in weather_files:
    os.remove(f)
print("🧹 Cleaned up individual weather files")
print(f"✅ Created: {output_file}")
