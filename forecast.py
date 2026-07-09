"""
=========================================================
Time Series Forecasting Project
Forecast Module
Author : Hania Eman
=========================================================
"""

# ==========================================================
# Import Libraries
# ==========================================================

import os
import pickle
import logging
import warnings

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# File Paths
# ==========================================================

MODEL_PATH = "models/sarima_model.pkl"
DATA_PATH = "data/business_time_series.csv"
OUTPUT_PATH = "outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# Check Files
# ==========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found! Please run train.py first."
    )

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found!"
    )

logging.info("Required files found.")

# ==========================================================
# Load Model
# ==========================================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

logging.info("SARIMA model loaded successfully.")

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df.set_index("Date", inplace=True)

print("\nDataset Loaded Successfully")
print(df.tail())

# ==========================================================
# Forecast Next 30 Days
# ==========================================================

forecast = model.get_forecast(steps=30)

forecast_values = forecast.predicted_mean

confidence_interval = forecast.conf_int()

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": forecast_values.values,
    "Lower CI": confidence_interval.iloc[:, 0].values,
    "Upper CI": confidence_interval.iloc[:, 1].values
})

print("\nFuture Forecast Preview")
print(forecast_df.head())

# ==========================================================
# Save Forecast CSV
# ==========================================================

forecast_df.to_csv(
    "outputs/forecast_results.csv",
    index=False
)

logging.info("Forecast CSV saved successfully.")

# ==========================================================
# Forecast Visualization
# ==========================================================

plt.figure(figsize=(15, 6))

# Historical Data
plt.plot(
    df.index,
    df["Sales"],
    label="Historical Sales",
    linewidth=2
)

# Forecast
plt.plot(
    future_dates,
    forecast_values,
    color="red",
    linewidth=2,
    label="Forecast"
)

# Confidence Interval
plt.fill_between(
    future_dates,
    confidence_interval.iloc[:, 0],
    confidence_interval.iloc[:, 1],
    color="orange",
    alpha=0.30,
    label="95% Confidence Interval"
)

plt.title("30-Day Sales Forecast (SARIMA)", fontsize=16)

plt.xlabel("Date", fontsize=12)

plt.ylabel("Sales", fontsize=12)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/forecast_plot.png",
    dpi=300
)

plt.close()

logging.info("Forecast graph saved.")

# ==========================================================
# Forecast Summary
# ==========================================================

print("\n" + "=" * 60)
print("FORECAST SUMMARY")
print("=" * 60)

print(f"Forecast Start Date : {future_dates[0].date()}")
print(f"Forecast End Date   : {future_dates[-1].date()}")

print("\nFirst 10 Predictions\n")

print(forecast_df.head(10))

# ==========================================================
# Export Forecast Report
# ==========================================================

summary_path = "outputs/forecast_summary.txt"

with open(summary_path, "w") as file:

    file.write("=====================================\n")
    file.write("SARIMA Forecast Report\n")
    file.write("=====================================\n\n")

    file.write(
        f"Forecast Start : {future_dates[0].date()}\n"
    )

    file.write(
        f"Forecast End   : {future_dates[-1].date()}\n\n"
    )

    file.write("Forecast Values\n\n")

    file.write(
        forecast_df.to_string(index=False)
    )

logging.info("Forecast summary exported.")

# ==========================================================
# Forecast Statistics
# ==========================================================

print("\nForecast Statistics\n")

print(forecast_df.describe())

# ==========================================================
# Completed
# ==========================================================

print("\n" + "=" * 60)
print("FORECAST COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Model Loaded Successfully")
print("30-Day Forecast Generated")
print("Confidence Interval Calculated")
print("Forecast CSV Saved")
print("Forecast Plot Saved")
print("Summary Report Saved")

print("=" * 60)