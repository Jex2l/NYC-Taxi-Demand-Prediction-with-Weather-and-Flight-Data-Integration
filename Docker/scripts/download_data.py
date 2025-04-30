import os
import requests
import pandas as pd
import zipfile
import glob
from time import sleep
import os
from tqdm import tqdm
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ========== CONFIG ========== #
years = list(range(2009, 2025))
months = list(range(1, 13))

# Base URLs
TAXI_YELLOW_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
TAXI_GREEN_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{ym}.parquet"
BTS_FLIGHT_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

# ========== PATHS ========== #
taxi_dir = os.path.join("data", "taxi")
flight_dir = os.path.join("data", "flight")
weather_dir = os.path.join("data", "weather")

os.makedirs(taxi_dir, exist_ok=True)
os.makedirs(flight_dir, exist_ok=True)
os.makedirs(weather_dir, exist_ok=True)

# ========== 1. TAXI DATA ========== #
taxi_dfs = []

for year in tqdm(years, desc="Taxi Years"):
    for month in tqdm(months, desc=f"Months for {year}", leave=False):
        ym = f"{year}-{month:02d}"

        for taxi_type in ["yellow", "green"]:
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
                            'trip_dropoff_datetime': 'dropoff_datetime'
                        })
                    else:
                        raise ValueError(f"{fname} is missing pickup/dropoff columns (yellow)")

                else:  # green taxi
                    if 'lpep_pickup_datetime' in df.columns:
                        df = df.rename(columns={
                            'lpep_pickup_datetime': 'pickup_datetime',
                            'lpep_dropoff_datetime': 'dropoff_datetime',
                            'pulocationid': 'pickup_location_id',
                            'dolocationid': 'dropoff_location_id'
                        })
                    else:
                        raise ValueError(f"{fname} is missing pickup/dropoff columns (green)")

                # ✅ Enforce required columns
                required_cols = ['pickup_datetime', 'dropoff_datetime']
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"{fname} is missing required column: {col}")

                optional_cols = ['pickup_location_id', 'dropoff_location_id']
                final_cols = required_cols + [col for col in optional_cols if col in df.columns]

                df = df[final_cols]
                df.dropna(inplace=True)

                taxi_dfs.append(df)
                print(f"✅ Processed {fname}")

            except Exception as e:
                print(f"❌ Failed to process {fname}: {e}")
                raise  # fail intentionally

if taxi_dfs:
    merged_taxi = pd.concat(taxi_dfs, ignore_index=True)
    output_file = os.path.join(taxi_dir, "all_taxi.csv")
    merged_taxi.to_csv(output_file, index=False)
    print(f"✅ Saved merged {output_file}")

    # 🧹 Cleanup
    for f in os.listdir(taxi_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(taxi_dir, f))
    print("🧹 Deleted individual taxi parquet files")
# ========== 2. FLIGHT DATA ========== #
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Setup session with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=10, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

flight_dfs = []

for year in tqdm(years,desc="Flight Years"):
    for month in tqdm(months,desc=f"Months for {year}", leave=False):
        ym = f"{year}_{month:02d}"
        zip_url = BTS_FLIGHT_URL.format(year=year, month=month)
        zip_path = os.path.join(flight_dir, f"flight_{ym}.zip")

        if not os.path.exists(zip_path):
            try:
                r = session.get(zip_url, timeout=(10, 120))  # (connect_timeout, read_timeout)
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    f.write(r.content)
                print(f"Downloaded {zip_path}")
            except Exception as e:
                print(f"Failed flight zip {ym}: {e}")
                continue

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(flight_dir)
            print(f"Extracted {zip_path}")
        except Exception as e:
            print(f"Failed extracting {zip_path}: {e}")
            continue

        time.sleep(10)  # prevent hammering BTS server

# Read all extracted flight CSVs
csvs = glob.glob(os.path.join(flight_dir, "*.csv"))
for csv_file in csvs:
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        flight_dfs.append(df)
    except Exception as e:
        print(f"Failed to read {csv_file}: {e}")

if flight_dfs:
    merged_flights = pd.concat(flight_dfs, ignore_index=True)
    output_file = os.path.join(flight_dir, "all_flights.csv")
    merged_flights.to_csv(output_file, index=False)
    print(f"Saved merged {output_file}")

    # Clean up
    for f in os.listdir(flight_dir):
        if f.endswith(".zip") or (f.endswith(".csv") and f != "all_flights.csv"):
            os.remove(os.path.join(flight_dir, f))
    print("🧹 Deleted individual flight files")

# ========== 3. WEATHER DATA ========== #
weather_dir = os.path.join("data", "weather")
os.makedirs(weather_dir, exist_ok=True)

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

for year in tqdm(years,desc="Weather Years"):
    for month in tqdm(months,desc=f"Months for {year}", leave=False):
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

# Merge and save
weather_dfs = []
for f in weather_files:
    try:
        df = pd.read_csv(f, skiprows=5)
        weather_dfs.append(df)
    except Exception as e:
        print(f"Could not read {f}: {e}")

if weather_dfs:
    merged_weather = pd.concat(weather_dfs, ignore_index=True)
    output_file = os.path.join(weather_dir, "all_weather.csv")
    merged_weather.to_csv(output_file, index=False)
    print(f"Saved merged weather to {output_file}")

    for f in weather_files:
        os.remove(f)
    print("🧹 Cleaned up individual weather files")
