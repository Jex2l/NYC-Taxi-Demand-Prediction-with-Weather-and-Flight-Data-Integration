import pandas as pd
import os

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
    taxi_path = os.path.join("data", "taxi", "taxi_features_full.csv")
    flight_path = os.path.join("data", "flight", "flight_features_full.csv")
    weather_path = os.path.join("data", "weather", "weather_features_full.csv")
    output_path = os.path.join("data", "output", "final_features.csv")

    merge_all(taxi_path, flight_path, weather_path, output_path)
