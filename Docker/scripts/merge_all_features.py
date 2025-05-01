import pandas as pd
import os
import sys

def merge_all(taxi, flight, weather, output_path):
    taxi_df = pd.read_csv(taxi)
    flight_df = pd.read_csv(flight)
    weather_df = pd.read_csv(weather)

    merged = pd.merge(
        taxi_df, flight_df,
        on=['location_id', 'year', 'month', 'day', 'hour', 'minute', 'dow'],
        how='left'
    ).fillna(0)

    final = pd.merge(
        merged, weather_df,
        on=['location_id', 'year', 'month', 'day', 'hour', 'minute', 'dow'],
        how='left'
    ).fillna(0)

    final.to_csv(output_path, index=False)
    print(f"Final merged features saved to {output_path}")

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    ym = f"{year}_{month:02d}"
    data_root = "/app/data"
    taxi_path = os.path.join(data_root, "taxi", f"taxi_features_{ym}.csv")
    flight_path = os.path.join(data_root, "flight", f"flight_features_{ym}.csv")
    weather_path = os.path.join(data_root, "weather", f"weather_features_{ym}.csv")
    output_path = os.path.join(data_root, "output", f"final_features_{ym}.csv")

    merge_all(taxi_path, flight_path, weather_path, output_path)
