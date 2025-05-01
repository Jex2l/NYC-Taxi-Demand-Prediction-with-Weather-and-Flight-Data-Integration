#!/bin/bash

run_etl() {
  YEAR=$1
  MONTH=$2

  echo -e "\n🚀 Starting ETL for $YEAR-$MONTH..."

  docker compose -f Docker/docker-compose-etl.yml run extract-data python3 download_data.py "$YEAR" "$MONTH"
  docker compose -f Docker/docker-compose-etl.yml run transform-taxi python3 prepare_taxi_data.py "$YEAR" "$MONTH"
  docker compose -f Docker/docker-compose-etl.yml run transform-flight python3 prepare_flight_data.py "$YEAR" "$MONTH"
  docker compose -f Docker/docker-compose-etl.yml run transform-weather python3 prepare_weather_data.py "$YEAR" "$MONTH"
  docker compose -f Docker/docker-compose-etl.yml run merge-features python3 merge_all_features.py "$YEAR" "$MONTH"
  docker compose -f Docker/docker-compose-etl.yml run load-data python3 upload_to_object_store.py "$YEAR" "$MONTH"
}

if [ $# -eq 2 ]; then
  # Run for a single month
  run_etl "$1" "$2"
else
  # Run from Jan 2015 to Dec 2024
  for year in {2015..2024}; do
    for month in {1..12}; do
      run_etl "$year" $(printf "%02d" "$month")
      if [ $? -ne 0 ]; then
        echo "❌ ETL failed for $year-$month. Exiting."
        exit 1
      fi
    done
  done
  echo -e "\n🎉 ETL completed for all months from 2015 to 2024!"
fi