import pandas as pd
import os
import sys
station_to_locationid = {
    'EWR': 1,
    'JFK': 132,
    'LGA': 138
}

def prepare_weather_data(path, output_path):
    df = pd.read_csv(path)
    df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
    df = df.dropna(subset=['station', 'valid'])
    df = df[df['station'].isin(station_to_locationid.keys())].copy()
    df['location_id'] = df['station'].map(station_to_locationid)
    df['hour_timestamp'] = df['valid'].dt.floor('h')

    keep_cols = ['hour_timestamp', 'location_id', 'tmpf', 'dwpf', 'relh', 'feel', 'sknt']
    df = df[keep_cols]
    df = df.groupby(['hour_timestamp', 'location_id'], as_index=False).mean()

    start_date = df['hour_timestamp'].min()
    end_date = df['hour_timestamp'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    full_grid = pd.MultiIndex.from_product(
        [date_range, [1, 132, 138]],
        names=['hour_timestamp', 'location_id']
    ).to_frame(index=False)

    df = full_grid.merge(df, on=['hour_timestamp', 'location_id'], how='left')
    df.sort_values(['location_id', 'hour_timestamp'], inplace=True)
    df = df.ffill()

    # Expand to 15-min intervals
    intervals = [0, 15, 30, 45]
    repeated = []
    for minute in intervals:
        temp = df.copy()
        temp['datetime'] = temp['hour_timestamp'] + pd.to_timedelta(minute, unit='m')
        temp['minute'] = minute
        repeated.append(temp)

    expanded = pd.concat(repeated, ignore_index=True)

    # Extract time keys
    expanded['year'] = expanded['datetime'].dt.year
    expanded['month'] = expanded['datetime'].dt.month
    expanded['day'] = expanded['datetime'].dt.day
    expanded['hour'] = expanded['datetime'].dt.hour
    expanded['dow'] = expanded['datetime'].dt.dayofweek

    final = expanded[['year', 'month', 'day', 'hour', 'minute', 'dow', 'location_id',
                      'tmpf', 'dwpf', 'relh', 'feel', 'sknt']]
    for col in ['tmpf', 'dwpf', 'relh', 'feel', 'sknt']:
        final.loc[:, col] = final[col].round(2)

    final.to_csv(output_path, index=False)
    print("Weather features saved to ", output_path)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    weather_path = os.path.join("data", "weather", f"all_weather_{year}_{month:02d}.csv")
    output_path = os.path.join("data", "weather", f"weather_features_{year}_{month:02d}.csv")
    prepare_weather_data(weather_path, output_path)
