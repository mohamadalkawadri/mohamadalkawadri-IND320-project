import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sidebar import setup_sidebar
setup_sidebar()

# ---------------------------------------------------------
# 1. SARIMAX Forecast Function (No Exogenous Variables)
# ---------------------------------------------------------
def run_sarimax_forecast(
    df,
    target_col,
    train_start,
    train_end,
    forecast_steps,
    order,
    seasonal_order,
    trend,
):
    """
    Fits SARIMAX on selected timeframe and makes forecasts.
    UNIVARIATE VERSION (no exogenous variables).
    """

    # Sort and set index
    df = df.sort_values("startTime").set_index("startTime")

    # Target series
    y = df[target_col].astype(float)

    # Training slice
    y_train = y.loc[train_start:train_end]

    # --- Fit SARIMAX model ---
    model = sm.tsa.statespace.SARIMAX(
        endog=y_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    # --- Forecast ---
    forecast_res = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()

    # -----------------------------------------------------
    # FIX: Give forecast a correct hourly datetime index
    # -----------------------------------------------------
    last_time = y_train.index[-1]

    forecast_index = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=forecast_steps,
        freq="H"
    )

    forecast_mean.index = forecast_index
    forecast_ci.index = forecast_index

    return results, forecast_mean, forecast_ci


# ---------------------------------------------------------
# 2. Plotting Function
# ---------------------------------------------------------
def plot_sarimax_forecast(
    y,
    forecast_mean,
    forecast_ci,
    train_end,
    title="SARIMAX Forecast"
):
    fig = go.Figure()

    # Observed data
    fig.add_trace(go.Scatter(
        x=y.index, y=y,
        name="Observed",
        mode="lines",
        line=dict(color="steelblue"),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_mean.index,
        y=forecast_mean.values,
        name="Forecast",
        mode="lines",
        line=dict(color="darkorange"),
    ))

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=forecast_mean.index.tolist() + forecast_mean.index[::-1].tolist(),
        y=forecast_ci.iloc[:, 0].tolist() + forecast_ci.iloc[:, 1][::-1].tolist(),
        fill="toself",
        fillcolor="rgba(255,165,0,0.2)",
        line=dict(color="rgba(255,165,0,0)"),
        hoverinfo="skip",
        name="Confidence Interval"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=600,
    )

    return fig


# ---------------------------------------------------------
# 3. Streamlit SARIMAX Interface (Univariate)
# ---------------------------------------------------------
def sarimax_interface(df: pd.DataFrame):

    st.header("🔮 SARIMAX Forecasting of Energy Production / Consumption")

    # Ensure datetime is timezone-naive
    df["startTime"] = pd.to_datetime(df["startTime"]).dt.tz_localize(None)
    df = df.sort_values("startTime")

    # ------------------------------
    # User Selections
    # ------------------------------
    st.subheader("Select Data")

    price_area = st.selectbox("Price Area:", sorted(df["priceArea"].unique()))
    production_group = st.selectbox("Production Group:", sorted(df["productionGroup"].unique()))

    df_filtered = df[
        (df["priceArea"] == price_area)
        & (df["productionGroup"] == production_group)
    ].copy()

    if df_filtered.empty:
        st.error("No data for selected area/group.")
        return

    target_col = "quantityKwh"

    # ------------------------------
    # Timeframe Selection
    # ------------------------------
    st.subheader("Training Timeframe")

    min_date = df_filtered["startTime"].min().to_pydatetime()
    max_date = df_filtered["startTime"].max().to_pydatetime()

    train_start_default = min_date
    train_end_default = max_date - pd.Timedelta(days=60)

    train_range = st.slider(
        "Select training period:",
        min_value=min_date,
        max_value=max_date,
        value=(train_start_default, train_end_default),
        format="YYYY-MM-DD"
    )
    train_start, train_end = train_range

    forecast_steps = st.number_input("Forecast Horizon (steps):", 1, 500, 48)

    # ------------------------------
    # SARIMAX Parameters
    # ------------------------------
    st.subheader("SARIMAX Parameters")

    p = st.number_input("p (AR order):", 0, 10, 1)
    d = st.number_input("d (Difference order):", 0, 2, 1)
    q = st.number_input("q (MA order):", 0, 10, 1)

    P = st.number_input("P (Seasonal AR):", 0, 10, 1)
    D = st.number_input("D (Seasonal diff):", 0, 2, 1)
    Q = st.number_input("Q (Seasonal MA):", 0, 10, 1)
    s = st.number_input("Seasonal period s:", 1, 500, 24)

    trend = st.selectbox("Trend:", ["n", "c", "t", "ct"])

    order = (p, d, q)
    seasonal_order = (P, D, Q, s)

    # ------------------------------
    # Run Model
    # ------------------------------
    if st.button("Run SARIMAX Forecast"):
        with st.spinner("Fitting SARIMAX model…"):
            results, forecast_mean, forecast_ci = run_sarimax_forecast(
                df_filtered,
                target_col,
                train_start,
                train_end,
                forecast_steps,
                order,
                seasonal_order,
                trend,
            )

        # Plot
        fig = plot_sarimax_forecast(
            y=df_filtered.set_index("startTime")[target_col],
            forecast_mean=forecast_mean,
            forecast_ci=forecast_ci,
            train_end=train_end,
            title=f"SARIMAX Forecast — {price_area} ({production_group})"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Summary")
        st.text(results.summary())


# ---------------------------------------------------------------------
# MongoDB loader
# ---------------------------------------------------------------------
from pymongo.mongo_client import MongoClient
from datetime import datetime

@st.cache_resource(show_spinner=False)
def get_mongo_collection():
    uri = st.secrets.get("MONGO_URI")
    client = MongoClient(uri, tlsAllowInvalidCertificates=True)
    database = client["assignment4"]
    collection = database["elhub"]
    return collection


@st.cache_data(show_spinner=False)
def load_energy(year: int) -> pd.DataFrame:
    coll = get_mongo_collection()

    query = {
        "startTime": {
            "$gte": datetime(year, 1, 1),
            "$lt": datetime(year + 1, 1, 1),
        }
    }

    cursor = coll.find(
        query,
        {
            "_id": 0,
            "startTime": 1,
            "priceArea": 1,
            "productionGroup": 1,
            "quantityKwh": 1,
        },
    )

    df = pd.DataFrame(list(cursor))

    if df.empty:
        st.error(f"MongoDB returned an empty dataset for year {year}.")
        st.stop()

    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["startTime"]).sort_values("startTime")

    return df


# -------------------------------------------------------
# Load and run interface
# -------------------------------------------------------
df = load_energy(st.session_state.year)
sarimax_interface(df)
