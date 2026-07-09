"""
=========================================================
Time Series Forecasting Project
Author : Hania Eman
Model  : SARIMA
=========================================================
"""

# =========================================================
# Import Required Libraries
# =========================================================

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# Create Required Directories
# =========================================================

folders = [
    "data",
    "models",
    "outputs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

logging.info("Project folders are ready.")

# =========================================================
# Generate Synthetic Business Time Series Dataset
# =========================================================

def generate_dataset():

    np.random.seed(42)

    dates = pd.date_range(
        start="2020-01-01",
        periods=730,
        freq="D"
    )

    trend = np.linspace(100, 300, 730)

    seasonality = 30 * np.sin(
        np.arange(730) * 2 * np.pi / 30
    )

    noise = np.random.normal(
        0,
        10,
        730
    )

    sales = trend + seasonality + noise

    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales
    })

    return df

# =========================================================
# Save Dataset Automatically
# =========================================================

dataset_path = "data/business_time_series.csv"

if not os.path.exists(dataset_path):

    df = generate_dataset()

    df.to_csv(
        dataset_path,
        index=False
    )

    logging.info("Dataset created successfully.")

else:

    logging.info("Dataset already exists.")

# =========================================================
# Load Dataset
# =========================================================

df = pd.read_csv(dataset_path)

logging.info("Dataset Loaded Successfully")

print(df.head())

print("\nShape :", df.shape)

print("\nColumns")

print(df.columns)

print("\nInformation")

print(df.info())

print("\nMissing Values")

print(df.isnull().sum())

# =========================================================
# Convert Date Column
# =========================================================

df["Date"] = pd.to_datetime(df["Date"])

df.set_index("Date", inplace=True)

logging.info("Date column converted.")

# =========================================================
# Basic Statistics
# =========================================================

print("\nStatistical Summary")

print(df.describe())

# =========================================================
# Plot Time Series
# =========================================================

plt.figure(figsize=(15,6))

plt.plot(
    df.index,
    df["Sales"],
    color="blue",
    linewidth=2
)

plt.title("Business Sales Over Time")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/time_series_plot.png")

plt.close()

logging.info("Time Series Plot Saved.")

# =========================================================
# Rolling Mean
# =========================================================

rolling_mean = df["Sales"].rolling(window=30).mean()

plt.figure(figsize=(15,6))

plt.plot(df["Sales"], label="Original")

plt.plot(
    rolling_mean,
    label="30-Day Rolling Mean"
)

plt.legend()

plt.title("Rolling Mean Analysis")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/rolling_mean.png")

plt.close()

logging.info("Rolling Mean Saved.")

print("\nTrain.py Part-1 Completed Successfully.")

# =========================================================
# Stationarity Test (ADF Test)
# =========================================================

from statsmodels.tsa.stattools import adfuller

print("\n" + "=" * 60)
print("ADF STATIONARITY TEST")
print("=" * 60)

adf_result = adfuller(df["Sales"])

print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"P-value       : {adf_result[1]:.4f}")

print("\nCritical Values")

for key, value in adf_result[4].items():
    print(f"{key} : {value:.4f}")

if adf_result[1] < 0.05:
    print("\nDataset is Stationary ✅")
else:
    print("\nDataset is NOT Stationary ❌")

# =========================================================
# Seasonal Decomposition
# =========================================================

from statsmodels.tsa.seasonal import seasonal_decompose

print("\nPerforming Seasonal Decomposition...")

decomposition = seasonal_decompose(
    df["Sales"],
    model="additive",
    period=30
)

fig = decomposition.plot()

fig.set_size_inches(14, 10)

plt.tight_layout()

plt.savefig("outputs/seasonal_decomposition.png")

plt.close()

logging.info("Seasonal decomposition saved.")

# =========================================================
# Seasonal Differencing
# =========================================================

print("\nApplying Seasonal Differencing...")

df["Seasonal_Diff"] = df["Sales"] - df["Sales"].shift(30)

df.dropna(inplace=True)

plt.figure(figsize=(15,6))

plt.plot(df.index, df["Seasonal_Diff"])

plt.title("Seasonally Differenced Series")

plt.xlabel("Date")

plt.ylabel("Differenced Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/seasonal_difference.png")

plt.close()

logging.info("Seasonal differencing completed.")

# =========================================================
# ADF Test After Differencing
# =========================================================

print("\n" + "=" * 60)
print("ADF TEST AFTER SEASONAL DIFFERENCING")
print("=" * 60)

adf_after = adfuller(df["Seasonal_Diff"])

print(f"ADF Statistic : {adf_after[0]:.4f}")
print(f"P-value       : {adf_after[1]:.4f}")

if adf_after[1] < 0.05:
    print("\nDataset is now Stationary ✅")
else:
    print("\nDataset is still NOT Stationary ❌")

# =========================================================
# ACF & PACF Plots
# =========================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12,6))
plot_acf(df["Seasonal_Diff"], lags=40)
plt.savefig("outputs/acf_plot.png")
plt.close()

plt.figure(figsize=(12,6))
plot_pacf(df["Seasonal_Diff"], lags=40, method="ywm")
plt.savefig("outputs/pacf_plot.png")
plt.close()

logging.info("ACF and PACF plots saved.")

# =========================================================
# Train-Test Split
# =========================================================

train_size = int(len(df) * 0.80)

train = df.iloc[:train_size]

test = df.iloc[train_size:]

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training Samples :", len(train))
print("Testing Samples  :", len(test))

logging.info("Train-test split completed.")

print("\nTrain.py  Completed Successfully.")

# =========================================================
# SARIMA MODEL TRAINING
# =========================================================

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

print("\n" + "=" * 60)
print("TRAINING SARIMA MODEL")
print("=" * 60)

model = SARIMAX(
    train["Sales"],
    order=(1,1,1),
    seasonal_order=(1,1,1,30),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_model = model.fit()

print("\nModel Training Completed Successfully!")

logging.info("SARIMA model trained successfully.")

# =========================================================
# SAVE MODEL
# =========================================================

model_path = "models/sarima_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(sarima_model, f)

print(f"\nModel saved at: {model_path}")

# =========================================================
# FORECAST TEST DATA
# =========================================================

forecast = sarima_model.get_forecast(
    steps=len(test)
)

predictions = forecast.predicted_mean

confidence = forecast.conf_int()

# =========================================================
# EVALUATION METRICS
# =========================================================

mae = mean_absolute_error(
    test["Sales"],
    predictions
)

rmse = mean_squared_error(
    test["Sales"],
    predictions
) ** 0.5

mape = (
    abs(
        (test["Sales"] - predictions)
        / test["Sales"]
    ).mean()
) * 100

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")

# Save metrics
with open("outputs/metrics.txt", "w") as file:
    file.write(f"MAE  : {mae:.2f}\n")
    file.write(f"RMSE : {rmse:.2f}\n")
    file.write(f"MAPE : {mape:.2f}%\n")

# =========================================================
# FORECAST PLOT
# =========================================================

plt.figure(figsize=(15,6))

plt.plot(
    train.index,
    train["Sales"],
    label="Training Data"
)

plt.plot(
    test.index,
    test["Sales"],
    label="Actual"
)

plt.plot(
    test.index,
    predictions,
    label="Forecast"
)

plt.fill_between(
    confidence.index,
    confidence.iloc[:,0],
    confidence.iloc[:,1],
    alpha=0.25
)

plt.title("SARIMA Forecast")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/forecast.png")

plt.close()

logging.info("Forecast plot saved.")

# =========================================================
# FUTURE 30-DAY FORECAST
# =========================================================

future = sarima_model.get_forecast(
    steps=30
)

future_predictions = future.predicted_mean

future_conf = future.conf_int()

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future_df = pd.DataFrame({

    "Date": future_dates,

    "Forecast": future_predictions.values,

    "Lower CI": future_conf.iloc[:,0].values,

    "Upper CI": future_conf.iloc[:,1].values

})

future_df.to_csv(
    "outputs/future_forecast.csv",
    index=False
)

print("\nFuture Forecast Saved Successfully!")

# =========================================================
# FUTURE FORECAST GRAPH
# =========================================================

plt.figure(figsize=(15,6))

plt.plot(
    df.index,
    df["Sales"],
    label="Historical Data"
)

plt.plot(
    future_dates,
    future_predictions,
    color="red",
    linewidth=2,
    label="Future Forecast"
)

plt.fill_between(
    future_dates,
    future_conf.iloc[:,0],
    future_conf.iloc[:,1],
    alpha=0.25
)

plt.title("30-Day Future Forecast")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/future_forecast.png")

plt.close()

logging.info("Future forecast saved.")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Model Saved")
print("Forecast Generated")
print("Metrics Calculated")
print("Future Predictions Exported")
print("Graphs Saved")
print("=" * 60)