"""
=========================================================
Model Interpretability using SHAP
Author : Hania Eman
=========================================================
"""

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import shap

warnings.filterwarnings("ignore")

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/business_time_series.csv")

df["Date"] = pd.to_datetime(df["Date"])

# =====================================================
# Create Lag Features
# =====================================================

df["Lag_1"] = df["Sales"].shift(1)
df["Lag_2"] = df["Sales"].shift(2)
df["Lag_3"] = df["Sales"].shift(3)
df["Lag_7"] = df["Sales"].shift(7)
df["Lag_30"] = df["Sales"].shift(30)

df.dropna(inplace=True)

# =====================================================
# Train Surrogate Model
# =====================================================

X = df[["Lag_1","Lag_2","Lag_3","Lag_7","Lag_30"]]

y = df["Sales"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

print("Random Forest trained successfully.")

# =====================================================
# SHAP Explainability
# =====================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# =====================================================
# SHAP Summary Plot
# =====================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_summary.png",
    dpi=300
)

plt.close()

# =====================================================
# SHAP Bar Plot
# =====================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap_feature_importance.png",
    dpi=300
)

plt.close()

print("SHAP graphs saved successfully.")

# =====================================================
# Feature Importance
# =====================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nFeature Importance\n")

print(importance)

print("\nInterpretability Completed Successfully.")