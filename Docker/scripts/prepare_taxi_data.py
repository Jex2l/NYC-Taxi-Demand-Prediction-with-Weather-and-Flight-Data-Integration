import pandas as pd
import os

def prepare_taxi_data(path, output_path):
    airport_ids = [1, 132, 138]

    taxi = pd.read_csv(path,low_memory=False)
    taxi.dropna(subset=['pickup_datetime', 'dropoff_datetime',
                        'pickup_location_id', 'dropoff_location_id'], inplace=True)

    # Ensure IDs are integers for filtering to work correctly
    taxi['pickup_location_id'] = taxi['pickup_location_id'].astype(int)
    taxi['dropoff_location_id'] = taxi['dropoff_location_id'].astype(int)

    # ✅ Filter to only trips involving JFK, LGA, EWR
    taxi = taxi[
        taxi['pickup_location_id'].isin(airport_ids) |
        taxi['dropoff_location_id'].isin(airport_ids)
    ].copy()

    # Convert datetimes and create 15-minute intervals
    taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])
    taxi['dropoff_datetime'] = pd.to_datetime(taxi['dropoff_datetime'])
    taxi['pickup_interval'] = taxi['pickup_datetime'].dt.floor('15min')
    taxi['dropoff_interval'] = taxi['dropoff_datetime'].dt.floor('15min')

    # --- Pickup Aggregation ---
    pickup = taxi[taxi['pickup_location_id'].isin(airport_ids)].copy()
    pickup['location_id'] = pickup['pickup_location_id']
    pickup['year'] = pickup['pickup_interval'].dt.year
    pickup['month'] = pickup['pickup_interval'].dt.month
    pickup['day'] = pickup['pickup_interval'].dt.day
    pickup['hour'] = pickup['pickup_interval'].dt.hour
    pickup['minute'] = pickup['pickup_interval'].dt.minute
    pickup['dow'] = pickup['pickup_interval'].dt.dayofweek
    pickup = pickup.groupby(['location_id', 'year', 'month', 'day', 'hour', 'minute', 'dow']) \
                   .size().reset_index(name='pickup_count')

    # --- Dropoff Aggregation ---
    dropoff = taxi[taxi['dropoff_location_id'].isin(airport_ids)].copy()
    dropoff['location_id'] = dropoff['dropoff_location_id']
    dropoff['year'] = dropoff['dropoff_interval'].dt.year
    dropoff['month'] = dropoff['dropoff_interval'].dt.month
    dropoff['day'] = dropoff['dropoff_interval'].dt.day
    dropoff['hour'] = dropoff['dropoff_interval'].dt.hour
    dropoff['minute'] = dropoff['dropoff_interval'].dt.minute
    dropoff['dow'] = dropoff['dropoff_interval'].dt.dayofweek
    dropoff = dropoff.groupby(['location_id', 'year', 'month', 'day', 'hour', 'minute', 'dow']) \
                     .size().reset_index(name='dropoff_count')

    # Merge and output
    taxi_volume = pd.merge(pickup, dropoff,
                           on=['location_id', 'year', 'month', 'day', 'hour', 'minute', 'dow'],
                           how='outer').fillna(0)

    taxi_volume['pickup_count'] = taxi_volume['pickup_count'].astype(int)
    taxi_volume['dropoff_count'] = taxi_volume['dropoff_count'].astype(int)

    taxi_volume.to_csv(output_path, index=False)
    print(f"Taxi features saved to {output_path}")

if __name__ == "__main__":
    taxi_path = os.path.join("data", "taxi", "all_taxi.csv")
    output_path = os.path.join("data", "taxi", "taxi_features_full.csv")
    prepare_taxi_data(taxi_path, output_path)
