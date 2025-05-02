import pandas as pd
import os
import sys
from datetime import datetime

def parse_time(t):
    if pd.isna(t):
        return pd.NaT
    try:
        t = int(float(t))  # safely handle '900.0', 900.0, '0900', etc.
    except ValueError:
        return pd.NaT
    t = str(t).zfill(4)     # pad to 4 digits
    return pd.to_timedelta(f"{t[:2]}:{t[2:]}:00")

def preprocess_flight_data(flight_df):
    nyc_airports = ['EWR', 'JFK', 'LGA']

    # Convert dates and times
    flight_df['FL_DATE'] = pd.to_datetime(flight_df['FL_DATE'])
    flight_df['dep_time'] = flight_df['FL_DATE'] + flight_df['CRS_DEP_TIME'].apply(parse_time)
    flight_df['arr_time'] = flight_df['FL_DATE'] + flight_df['CRS_ARR_TIME'].apply(parse_time)

    # Round to 15-min intervals
    flight_df['dep_interval'] = flight_df['dep_time'].dt.floor('15min')
    flight_df['arr_interval'] = flight_df['arr_time'].dt.floor('15min')

    # Filter for NYC airports
    departures = flight_df[flight_df['ORIGIN'].isin(nyc_airports)][['dep_interval', 'ORIGIN']]
    arrivals = flight_df[flight_df['DEST'].isin(nyc_airports)][['arr_interval', 'DEST']]

    # Create base grid
    all_times = pd.date_range(
        start=flight_df['FL_DATE'].min(),
        end=flight_df['FL_DATE'].max() + pd.Timedelta('2h'),
        freq='15min'
    )
    base = pd.MultiIndex.from_product([all_times, nyc_airports], names=['interval', 'airport']).to_frame(index=False)

    # Aggregate departures and arrivals
    dep_counts = departures.groupby(['dep_interval', 'ORIGIN']).size().reset_index(name='dep_now')
    arr_counts = arrivals.groupby(['arr_interval', 'DEST']).size().reset_index(name='arr_now')
    dep_counts.columns = ['interval', 'airport', 'dep_now']
    arr_counts.columns = ['interval', 'airport', 'arr_now']

    flight_features = base.merge(dep_counts, on=['interval', 'airport'], how='left') \
                          .merge(arr_counts, on=['interval', 'airport'], how='left') \
                          .fillna(0)

    # Create future window features
    for minutes in [30, 60, 90, 120]:
        delta = pd.Timedelta(minutes=minutes)

        temp = flight_features[['interval', 'airport', 'dep_now']].copy()
        temp['interval'] = temp['interval'] - delta
        temp.columns = ['interval', 'airport', f'dep_next_{minutes}']
        flight_features = flight_features.merge(temp, on=['interval', 'airport'], how='left')

        temp = flight_features[['interval', 'airport', 'arr_now']].copy()
        temp['interval'] = temp['interval'] - delta
        temp.columns = ['interval', 'airport', f'arr_next_{minutes}']
        flight_features = flight_features.merge(temp, on=['interval', 'airport'], how='left')

    flight_features.fillna(0, inplace=True)

    # Extract keys
    flight_features['year'] = flight_features['interval'].dt.year
    flight_features['month'] = flight_features['interval'].dt.month
    flight_features['day'] = flight_features['interval'].dt.day
    flight_features['hour'] = flight_features['interval'].dt.hour
    flight_features['minute'] = flight_features['interval'].dt.minute
    flight_features['dow'] = flight_features['interval'].dt.dayofweek

    # Map airport to location ID
    airport_to_locid = {'EWR': 1, 'JFK': 132, 'LGA': 138}
    flight_features['location_id'] = flight_features['airport'].map(airport_to_locid)

    # Final columns
    keep_cols = [
        'year', 'month', 'day', 'hour', 'minute', 'dow', 'location_id',
        'dep_now', 'dep_next_30', 'dep_next_60', 'dep_next_90', 'dep_next_120',
        'arr_now', 'arr_next_30', 'arr_next_60', 'arr_next_90', 'arr_next_120'
    ]
    return flight_features[keep_cols]
if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    data_root = "/app/data"
    input_path = os.path.join(data_root , "flight", f"all_flights_{year}_{month:02d}.csv")
    output_path = os.path.join(data_root , "flight", f"flight_features_{year}_{month:02d}.csv")

    df = pd.read_csv(input_path, low_memory=False , on_bad_lines='skip')
    df.columns = [col.upper() for col in df.columns]
    if 'FLIGHTDATE' in df.columns:
        df.rename(columns={'FLIGHTDATE': 'FL_DATE'}, inplace=True)
    elif 'FL_DATE' not in df.columns:
        raise ValueError("❌ FL_DATE column not found in flight data.")

# Rename only if expected columns exist
    rename_map = {
        'CRSDEPTIME': 'CRS_DEP_TIME',
        'CRSARRTIME': 'CRS_ARR_TIME',
        'ORIGIN': 'ORIGIN',
        'DEST': 'DEST'
    }
    df.rename(columns=rename_map, inplace=True)
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df[(df['FL_DATE'].dt.year == year) & (df['FL_DATE'].dt.month == month)]

    if df.empty:
        print(f"❌ No data for {year}-{month:02d} in flight input.")
        sys.exit(1)

    result = preprocess_flight_data(df)
    result.to_csv(output_path, index=False)
    print(f"✅ Flight features saved to {output_path} ({len(result)} rows)")