"""
=========================================================
Time Series Forecasting Dashboard
Author : Hania Eman
=========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# WELCOME ANIMATION
# ==========================================================

if "first_visit" not in st.session_state:
    st.session_state.first_visit = True
    st.balloons()
    st.toast("Welcome to the Dashboard 🎉")

# ==========================================================
# LOAD DATASET
# ==========================================================

DATA_PATH = "data/business_time_series.csv"

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Hide only menu and footer */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

/* Background */

.stApp{
background:linear-gradient(
135deg,
#F4F8FC,
#EEF5FF,
#FFFFFF
);
}

/* Hero Banner */

.hero{

background:linear-gradient(
90deg,
#2563EB,
#3B82F6,
#60A5FA
);

padding:40px;

border-radius:20px;

text-align:center;

color:white;

box-shadow:0px 10px 25px rgba(0,0,0,.15);

margin-bottom:25px;

}

/* Metric Cards */

[data-testid="metric-container"]{

background:white;

padding:18px;

border-radius:18px;

border-left:5px solid #2563EB;

box-shadow:0px 5px 15px rgba(0,0,0,.10);

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#F7FBFF;

}

/* Buttons */

.stButton>button{

background:#2563EB;

color:white;

border-radius:10px;

border:none;

height:45px;

width:100%;

font-weight:bold;

}

.stButton>button:hover{

background:#1D4ED8;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
"https://img.icons8.com/fluency/96/combo-chart.png",
width=80
)

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(

"Choose a Page",

[
    "🏠 Dashboard",
    "📊 Dataset",
    "📈 Historical Analysis",
    "🔮 Forecast",
    "🧠 Explainability",
    "📥 Downloads",
    "⚙ Settings",
    "👩 About"
]

)

st.sidebar.markdown("---")

st.sidebar.success("Dashboard Ready ✅")

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Dashboard":

    current_time = datetime.now().strftime("%d %B %Y | %I:%M %p")

    st.markdown(f"""
    <div class="hero">

    <h1>📈 Time Series Forecasting Dashboard</h1>

    <h2>Welcome Hania 👋</h2>

    <p><b>{current_time}</b></p>

    <h4>Forecast Tomorrow with Confidence</h4>

    <p>SARIMA • SHAP • Streamlit • Plotly</p>

    </div>
    """, unsafe_allow_html=True)

    st.success("🎉 Welcome! Your dashboard is ready.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📄 Total Records", len(df))
    c2.metric("📈 Average Sales", f"{df['Sales'].mean():.2f}")
    c3.metric("🚀 Maximum Sales", f"{df['Sales'].max():.2f}")
    c4.metric("📉 Minimum Sales", f"{df['Sales'].min():.2f}")

    st.write("")

    left, right = st.columns([2,1])

    with left:

        fig = px.line(
            df,
            x="Date",
            y="Sales",
            markers=True,
            title="Historical Business Sales"
        )

        fig.update_layout(
            template="plotly_white",
            height=520
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("🚀 Dashboard Features")

        st.markdown("""
- ✅ Modern UI
- ✅ SARIMA Forecasting
- ✅ SHAP Explainability
- ✅ Interactive Charts
- ✅ KPI Analytics
- ✅ Download Center
- ✅ Portfolio Ready
        """)

        st.info("""
**Project Objective**

Predict future business sales using Time Series Forecasting and explain model predictions with SHAP.
""")
        # ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    st.markdown("Explore the complete business sales dataset.")

    col1, col2 = st.columns(2)

    with col1:
        search = st.text_input(
            "🔍 Search by Sales Value",
            placeholder="Example: 150"
        )

    with col2:
        rows = st.selectbox(
            "Rows to Display",
            [10, 20, 50, 100],
            index=1
        )

    dataset = df.copy()

    if search:
        dataset = dataset[
            dataset["Sales"].astype(str).str.contains(search)
        ]

    st.dataframe(
        dataset.head(rows),
        use_container_width=True,
        height=450
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ==========================================================
# HISTORICAL ANALYSIS
# ==========================================================

elif page == "📈 Historical Analysis":

    st.title("📈 Historical Sales Analysis")

    tab1, tab2, tab3 = st.tabs(
        [
            "Trend",
            "Distribution",
            "Monthly Analysis"
        ]
    )

    # ------------------------------------------------------

    with tab1:

        trend = px.line(
            df,
            x="Date",
            y="Sales",
            markers=True,
            title="Business Sales Trend"
        )

        trend.update_layout(
            template="plotly_white",
            height=520
        )

        st.plotly_chart(
            trend,
            use_container_width=True
        )

    # ------------------------------------------------------

    with tab2:

        col1, col2 = st.columns(2)

        with col1:

            hist = px.histogram(
                df,
                x="Sales",
                nbins=40,
                title="Sales Distribution"
            )

            hist.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                hist,
                use_container_width=True
            )

        with col2:

            box = px.box(
                df,
                y="Sales",
                title="Sales Box Plot"
            )

            box.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                box,
                use_container_width=True
            )

    # ------------------------------------------------------

    with tab3:

        monthly = df.copy()

        monthly["Month"] = monthly["Date"].dt.strftime("%b")

        monthly_sales = (
            monthly
            .groupby("Month")["Sales"]
            .mean()
            .reset_index()
        )

        month_order = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ]

        monthly_sales["Month"] = pd.Categorical(
            monthly_sales["Month"],
            categories=month_order,
            ordered=True
        )

        monthly_sales = monthly_sales.sort_values("Month")

        monthly_chart = px.bar(
            monthly_sales,
            x="Month",
            y="Sales",
            title="Average Monthly Sales"
        )

        monthly_chart.update_layout(
            template="plotly_white",
            height=500
        )

        st.plotly_chart(
            monthly_chart,
            use_container_width=True
        )

        st.dataframe(
            monthly_sales,
            use_container_width=True
        )
        # ==========================================================
# FORECAST PAGE
# ==========================================================

elif page == "🔮 Forecast":

    st.title("🔮 Future Sales Forecast")

    forecast_path = "outputs/forecast_results.csv"

    if os.path.exists(forecast_path):

        forecast_df = pd.read_csv(forecast_path)
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Sales"],
                mode="lines",
                name="Historical Sales",
                line=dict(color="#2563EB", width=3)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#EF4444", width=3)
            )
        )

        fig.update_layout(
            template="plotly_white",
            height=550,
            title="Historical vs Forecast Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Results")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        st.download_button(
            "⬇ Download Forecast CSV",
            forecast_df.to_csv(index=False),
            "forecast_results.csv",
            "text/csv"
        )

    else:

        st.warning("⚠ Run forecast.py first.")

# ==========================================================
# EXPLAINABILITY PAGE
# ==========================================================

elif page == "🧠 Explainability":

    st.title("🧠 Model Explainability")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("SHAP Summary")

        if os.path.exists("outputs/shap_summary.png"):

            st.image(
                "outputs/shap_summary.png",
                use_container_width=True
            )

        else:

            st.info("SHAP Summary not available.")

    with col2:

        st.subheader("Feature Importance")

        if os.path.exists("outputs/shap_feature_importance.png"):

            st.image(
                "outputs/shap_feature_importance.png",
                use_container_width=True
            )

        else:

            st.info("Feature Importance plot not available.")

    importance = "outputs/feature_importance.csv"

    if os.path.exists(importance):

        st.subheader("Feature Importance Table")

        feature_df = pd.read_csv(importance)

        st.dataframe(
            feature_df,
            use_container_width=True
        )

# ==========================================================
# DOWNLOAD PAGE
# ==========================================================

elif page == "📥 Downloads":

    st.title("📥 Download Center")

    files = [
        "outputs/forecast_results.csv",
        "outputs/feature_importance.csv",
        "outputs/metrics.txt"
    ]

    for file in files:

        if os.path.exists(file):

            with open(file, "rb") as f:

                st.download_button(
                    label=f"⬇ {os.path.basename(file)}",
                    data=f,
                    file_name=os.path.basename(file)
                )

# ==========================================================
# SETTINGS PAGE
# ==========================================================

elif page == "⚙ Settings":

    st.title("⚙ Dashboard Settings")

    theme = st.selectbox(
        "Select Theme",
        ["Professional", "Light", "Dark"]
    )

    horizon = st.slider(
        "Forecast Horizon",
        7,
        90,
        30
    )

    animation = st.toggle(
        "Enable Animation",
        value=True
    )

    st.success("Settings Updated Successfully.")

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "👩 About":

    st.title("👩 About This Project")

    st.markdown("""

## 📈 Time Series Forecasting Dashboard

A professional Machine Learning application developed using
**SARIMA (Seasonal ARIMA)** for forecasting future business sales.

### 🚀 Features

- Historical Trend Analysis
- Time Series Forecasting
- Interactive Dashboard
- SHAP Explainability
- Forecast Downloads
- KPI Analytics
- Professional Streamlit UI

### 🛠 Technology Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Statsmodels
- SHAP

### 👩‍💻 Developed By

**Hania Eman**

AI • Machine Learning • Data Science

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:gray;padding:15px;">

<b>📈 Time Series Forecasting Dashboard</b>

<br><br>

Developed with ❤️ using Python • Streamlit • Plotly • SARIMA

<br><br>

© 2026 Hania Eman | All Rights Reserved

</div>
""",
unsafe_allow_html=True
)