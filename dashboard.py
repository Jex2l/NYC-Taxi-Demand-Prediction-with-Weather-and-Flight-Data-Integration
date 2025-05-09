#!/usr/bin/env python3
import os
import glob
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# ── 1. Load Data ─────────────────────────────────────────────────
# Assumes you have NYC taxi zone GeoJSON in ./data/taxi_zones.geojson
zones = gpd.read_file("data/taxi_zones.geojson")  # TLC shape file
df_files = sorted(glob.glob("data/final_features_*.csv"))

def load_latest(n=1):
    """Load the N most recent monthly files."""
    files = df_files[-n:]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # aggregate by zone + hour
    agg = (df.groupby(["pickup_borough", "pickup_zone", "hour"])
             .pickup_count.sum()
             .reset_index())
    return agg

# ── 2. Build the App ─────────────────────────────────────────────
app = Dash(__name__)
app.layout = html.Div([
    html.H1("NYC Airport Taxi Demand Dashboard"),
    html.Div([
        html.Label("Months to load:"),
        dcc.Slider(id="months-slider", min=1, max=6, step=1, value=1,
                   marks={i: f"{i}" for i in range(1,7)}),
    ], style={"width":"50%", "padding":"20px"}),
    dcc.Graph(id="heatmap-graph"),
    dcc.Graph(id="time-series")
])

# ── 3. Callbacks ─────────────────────────────────────────────────
@app.callback(
    Output("heatmap-graph", "figure"),
    Output("time-series", "figure"),
    Input("months-slider", "value")
)
def update_dashboard(months):
    df = load_latest(months)
    # merge geometry
    gdf = zones.merge(df, left_on="zone_name", right_on="pickup_zone")

    # choropleth: average hourly demand per zone
    fig_map = px.choropleth_mapbox(
        gdf, geojson=gdf.geometry, locations=gdf.index,
        color="pickup_count",
        mapbox_style="carto-positron",
        zoom=9, center={"lat":40.7128,"lon":-74.0060},
        opacity=0.6,
        hover_data=["pickup_zone","pickup_count"]
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # time series: total demand by hour of day
    df_ts = df.groupby("hour").pickup_count.sum().reset_index()
    fig_ts = px.bar(df_ts, x="hour", y="pickup_count",
                    labels={"pickup_count":"Total pick‑ups","hour":"Hour of day"},
                    title="Hourly Demand (total)")

    return fig_map, fig_ts

# ── 4. Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)
