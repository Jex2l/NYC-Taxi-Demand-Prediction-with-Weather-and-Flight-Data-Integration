import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_predictions_from_s3, load_predictions_from_local
import os

st.title("🚕 NYC Taxi Demand Prediction Dashboard")

# --- Load predictions
source = st.sidebar.selectbox("Prediction Data Source", ["Local", "S3"])

if source == "Local":
    df = load_predictions_from_local("data/predictions")
else:
    df = load_predictions_from_s3(
        bucket="object-persist-project40",
        prefix="predictions",
        endpoint="https://chi.tacc.chameleoncloud.org:8080",
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

if df.empty:
    st.warning("No prediction data found.")
    st.stop()

# --- Select Month
df["month_str"] = df["month"].astype(str).str.zfill(2)
month_options = sorted(df["year"].astype(str) + "-" + df["month_str"].astype(str).unique())
selected_month = st.selectbox("Select Month", month_options)

filtered = df[(df["year"].astype(str) + "-" + df["month_str"]) == selected_month]

st.subheader(f"📈 Error Metrics for {selected_month}")
st.metric("RMSE", round((filtered["errors"].apply(lambda x: x["pickup_error"]**2).mean())**0.5, 2))
st.metric("MAE", round(filtered["errors"].apply(lambda x: abs(x["pickup_error"])).mean(), 2))

# --- Time Series Chart
st.subheader("📊 Pickup Count: Predicted vs Actual")
filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], format="%Y%m%d_%H%M%S")
filtered = filtered.sort_values("timestamp")

plt.figure(figsize=(12, 5))
plt.plot(filtered["timestamp"], filtered["prediction"].apply(lambda x: x["pickup_count"]), label="Predicted")
plt.plot(filtered["timestamp"], filtered["actual"].apply(lambda x: x["pickup_count"]), label="Actual")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Pickup Count")
plt.title("Pickup Count Over Time")
st.pyplot(plt)

# --- Data Quality
st.subheader("⚠️ Data Quality Checks")
bad_data = filtered[filtered["errors"].apply(lambda e: abs(e["pickup_error_pct"]) > 100)]
st.write(f"{len(bad_data)} rows with >100% prediction error.")
st.dataframe(bad_data[["timestamp", "prediction", "actual", "errors"]].head(10))
