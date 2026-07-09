# 📈 Time Series Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SARIMA](https://img.shields.io/badge/Forecasting-SARIMA-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

An end-to-end **Time Series Forecasting Dashboard** developed using **Python, SARIMA, Streamlit, Plotly, and SHAP** to analyze historical business sales data, forecast future trends, and improve model interpretability through interactive visualizations.

---

# 📌 Project Overview

Time Series Forecasting plays an important role in business planning by helping organizations estimate future trends based on historical data. This project demonstrates a complete forecasting workflow, starting from data preprocessing and exploratory analysis to model training, future prediction, and explainability.

The forecasting model is built using **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** to capture trend and seasonality within historical sales data. To improve model transparency, **SHAP (SHapley Additive Explanations)** is integrated to highlight feature contributions and support better interpretation of forecasting results.

The project also includes a responsive **Streamlit dashboard**, allowing users to explore datasets, visualize trends, generate forecasts, review model insights, and download forecasting results through an interactive interface.

---

# 🚀 Key Features

- Historical Time Series Analysis
- Data Preprocessing and Cleaning
- Stationarity Testing (ADF Test)
- Seasonal Decomposition
- Seasonal Differencing
- SARIMA Forecasting Model
- Future Sales Prediction
- Forecast Performance Evaluation
- SHAP-Based Model Explainability
- Interactive Streamlit Dashboard
- Interactive Plotly Visualizations
- Downloadable Forecast Reports
- Organized Project Structure

---

# 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Plotly |
| Forecasting | Statsmodels (SARIMA) |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Development Tools | VS Code, Jupyter Notebook |

---
# 📂 Project Structure

```text
Time-Series-Forecasting-Dashboard/
│
├── app.py                          # Streamlit dashboard
├── train.py                        # Model training pipeline
├── forecast.py                     # Future forecasting
├── interpretability.py             # SHAP explainability
├── business_time_series.csv        # Input dataset
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
│
├── outputs/
│   ├── time_series_plot.png
│   ├── rolling_mean.png
│   ├── seasonal_difference.png
│   ├── seasonal_decomposition.png
│   ├── acf_plot.png
│   ├── pacf_plot.png
│   ├── forecast_plot.png
│   ├── future_forecast.png
│   ├── forecast_results.csv
│   ├── future_forecast.csv
│   ├── metrics.txt
│   ├── forecast_summary.txt
│   ├── shap_summary.png
│   ├── shap_feature_importance.png
│   └── feature_importance.csv
│
└── models/
    └── sarima_model.pkl
```

---

# 📊 Project Workflow

The project follows a structured forecasting workflow:

1. Load the historical business sales dataset.
2. Perform data cleaning and preprocessing.
3. Conduct exploratory data analysis (EDA).
4. Visualize historical sales trends.
5. Check stationarity using the Augmented Dickey-Fuller (ADF) Test.
6. Apply seasonal differencing where required.
7. Train the SARIMA forecasting model.
8. Evaluate forecasting performance using standard metrics.
9. Generate future sales forecasts.
10. Interpret model predictions using SHAP.
11. Display results through an interactive Streamlit dashboard.
12. Export forecast reports, visualizations, and evaluation metrics.

---

# 📈 Forecasting Model

The forecasting pipeline is built using the **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** model, which is well-suited for time series data containing both trend and seasonal patterns.

The trained model is used to forecast future business sales while preserving seasonal behavior and long-term trends observed in the historical dataset.

---

# 🧠 Model Explainability

To improve model transparency, this project integrates **SHAP (SHapley Additive Explanations)** for explainability.

The explainability module helps users understand how features contribute to model predictions by generating:

- SHAP Summary Plot
- SHAP Feature Importance Plot
- Feature Importance Report
- Feature Importance CSV

These visualizations make the forecasting process more interpretable and easier to analyze.

---

# 📉 Model Evaluation

The forecasting model is evaluated using the following performance metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

These metrics help measure forecasting accuracy and overall model performance.

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/haniaeman2026-pixel/Time-Series-Forecasting-Dashboard.git
```

Navigate to the project directory

```bash
cd Time-Series-Forecasting-Dashboard
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train the Forecasting Model

```bash
python train.py
```

### Generate Future Forecast

```bash
python forecast.py
```

### Generate SHAP Explainability

```bash
python interpretability.py
```

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

After running the application, open your browser and visit:

```text
http://localhost:8501
```

---

# 🌐 Streamlit Dashboard

The project includes an interactive Streamlit dashboard that provides a simple and user-friendly interface for exploring forecasting results.

### Dashboard Modules

- 🏠 Dashboard
- 📊 Dataset Explorer
- 📈 Historical Analysis
- 🔮 Future Forecast
- 🧠 Model Explainability
- 📥 Download Center
- ⚙️ Settings
- 👩 About

---

# 📊 Dashboard Highlights

The dashboard provides:

- Interactive KPI Cards
- Historical Sales Visualization
- Forecast Comparison Charts
- SHAP Explainability
- Feature Importance Analysis
- Forecast Result Table
- CSV Download Support
- Interactive Plotly Charts

---

# 📷 Project Outputs

The project automatically generates the following outputs:

### Visualizations

- Time Series Plot
- Rolling Mean Plot
- Seasonal Decomposition
- Seasonal Difference Plot
- ACF Plot
- PACF Plot
- Forecast Plot
- Future Forecast Plot
- SHAP Summary Plot
- SHAP Feature Importance Plot

### Reports

- Forecast Results (CSV)
- Future Forecast (CSV)
- Forecast Summary
- Performance Metrics
- Feature Importance Report

---

# 📸 Dashboard Preview

After running the Streamlit application, add a screenshot of your dashboard here.

```text
assets/dashboard_preview.png
```

> **Tip:** Upload a screenshot of your dashboard to an `assets/` folder and replace the placeholder above with:

```markdown
![Dashboard Preview](assets/dashboard_preview.png)
```
---

# 💡 Future Improvements

The project can be extended with additional features, including:

- Support for multiple forecasting models (ARIMA, Prophet, LSTM)
- User-uploaded datasets
- Automatic hyperparameter tuning
- Real-time forecasting
- Cloud deployment using Streamlit Community Cloud
- REST API integration with FastAPI
- Interactive forecasting controls
- Advanced dashboard analytics
- Automated model retraining
- Docker containerization

---

# 🤝 Contributing

Contributions are welcome!

If you would like to improve this project, feel free to:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

Suggestions, improvements, and feedback are always appreciated.

---

# 👩‍💻 Author

## Hania Eman

Aspiring Artificial Intelligence, Machine Learning, and Data Science Engineer

### Connect with Me

- 💼 LinkedIn: *Add your LinkedIn profile*
- 💻 GitHub: https://github.com/haniaeman2026-pixel

---

# 📄 License

This project is developed for educational purposes, portfolio development, and internship submissions.

You are welcome to explore, learn from, and build upon this project with appropriate attribution.

---

# 🙏 Acknowledgements

Special thanks to the open-source community and the developers of the following libraries and frameworks:

- Python
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- SHAP
- Plotly
- Streamlit

Their excellent tools made this project possible.

---

# ⭐ Support

If you found this project helpful or interesting:

⭐ Star this repository

🍴 Fork the project

📝 Share your feedback

Your support is greatly appreciated.

---

## 📌 Project Summary

**Time Series Forecasting Dashboard** is a complete forecasting application that combines statistical modeling, data visualization, model interpretability, and an interactive web interface. It demonstrates practical implementation of time series forecasting using SARIMA while providing clear insights through SHAP explainability and a user-friendly Streamlit dashboard.

This project reflects best practices in data analysis, forecasting, visualization, and application development, making it suitable for learning, portfolio presentation, and internship showcases.

---

