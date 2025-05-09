#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# ── 1. Base and Data Paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "data"

# ── 2. Load Taxi Zone Geometry ──────────────────────────────────────
geojson_path = DATA_DIR / "taxi_zones.geojson"
if not geojson_path.exists():
    raise FileNotFoundError(f"GeoJSON not found at {geojson_path}")
zones = gpd.read_file(str(geojson_path))

# ── 3. Discover Feature CSVs ────────────────────────────────────────
pattern = str(DATA_DIR / "final_features_*.csv")
df_files = sorted(glob.glob(pattern))
if not df_files:
    raise FileNotFoundError(f"No feature CSVs found with pattern {pattern}")

# ── 4. Data Loading Function ─────────────────────────────────────────
def load_latest(months=1):
    files = df_files[-months:]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # aggregate total pickups by zone and hour
    agg = (
        df.groupby(["pickup_zone", "hour"])['pickup_count']
          .sum()
          .reset_index()
    )
    return agg

# ── 5. Build Dash App ────────────────────────────────────────────────
app = Dash(__name__)
app.layout = html.Div([
    html.H1("NYC Airport Taxi Demand Dashboard"),
    html.Div([
        html.Label("Months to load:"),
        dcc.Slider(
            id="months-slider",
            min=1,
            max=min(6, len(df_files)),
            step=1,
            value=1,
            marks={i: str(i) for i in range(1, min(6, len(df_files))+1)}
        ),
    ], style={"width": "50%", "padding": "20px"}),
    dcc.Graph(id="heatmap-graph"),
    dcc.Graph(id="time-series-graph")
])

# ── 6. Callbacks ─────────────────────────────────────────────────────
@app.callback(
    Output("heatmap-graph", "figure"),
    Output("time-series-graph", "figure"),
    Input("months-slider", "value")
)
def update_dashboard(months):
    df = load_latest(months)
    # merge shapefile by zone
    gdf = zones.merge(df, left_on="zone_name", right_on="pickup_zone")
    # choropleth map
    fig_map = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color="pickup_count",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 40.7128, "lon": -74.0060},
        opacity=0.6,
        hover_data=["pickup_zone", "pickup_count"]
    )
    fig_map.update_layout(margin={"r":0, "t":0, "l":0, "b":0})

    # time-series bar chart
    df_ts = df.groupby("hour")['pickup_count'].sum().reset_index()
    fig_ts = px.bar(
        df_ts,
        x="hour",
        y="pickup_count",
        labels={"pickup_count": "Total Pickups", "hour": "Hour of Day"},
        title="Hourly Demand Summary"
    )
    return fig_map, fig_ts

# ── 7. Run Server ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)

