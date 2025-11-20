import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sidebar import setup_sidebar
from pymongo.mongo_client import MongoClient
from datetime import datetime

setup_sidebar()

# ---------------------------------------------------------------------
# 1. SARIMAX Forecast Function (WITH Optional Exogenous Variables)
# ---------------------------------------------------------------------
def run_sarimax_forecast(
    df,
    target_col,
    exog_cols,
    train_start,
    train_end,
    forecast_steps,
    order,
    seasonal_order,
    trend,
):

    df = df.sort_values("startTime").set_index("startTime")

    # Target series
    y = df[target_col].astype(float)

    # Exogenous matrix
    X = None
    if exog_cols:
        X = df[exog_cols].astype(float)

    # Training slice
    y_train = y.loc[train_start:train_end]
    X_train = X.loc[train_start:train_end] if X is not None else None

    # ----------------------------------------
    # Fit SARIMAX
    # ----------------------------------------
    model = sm.tsa.statespace.SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    # ----------------------------------------
    # Build future exogenous forecast matrix
    # ----------------------------------------
    # ----------------------------------------
    if exog_cols:

        # Use asof to find nearest timestamp <= train_end
        nearest_time = X.index.asof(train_end)

        if pd.isna(nearest_time):
            nearest_time = X.index[0]

        last_vals = X.loc[nearest_time, exog_cols]

        # Repeat last values for all forecast steps
        X_future = pd.DataFrame(
            [last_vals.values] * forecast_steps,
            columns=exog_cols
        )
    else:
        X_future = None


    # ----------------------------------------
    # Forecast
    # ----------------------------------------
    forecast_res = results.get_forecast(steps=forecast_steps, exog=X_future)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()

    # Assign correct datetime index
    last_time = y_train.index[-1]
    forecast_index = pd.date_range(last_time + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")

    forecast_mean.index = forecast_index
    forecast_ci.index = forecast_index

    return results, forecast_mean, forecast_ci


# ---------------------------------------------------------------------
# 2. Plot Function
# ---------------------------------------------------------------------
def plot_sarimax_forecast(y, forecast_mean, forecast_ci, train_end, title):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y.index, y=y,
        mode="lines",
        name="Observed",
        line=dict(color="steelblue")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_mean.index,
        y=forecast_mean,
        mode="lines",
        name="Forecast",
        line=dict(color="darkorange")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_mean.index.tolist() + forecast_mean.index[::-1].tolist(),
        y=forecast_ci.iloc[:, 0].tolist() + forecast_ci.iloc[:, 1][::-1].tolist(),
        fill="toself",
        fillcolor="rgba(255,165,0,0.2)",
        name="Confidence Interval",
        hoverinfo="skip",
        line=dict(color="rgba(255,165,0,0)")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        height=650,
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# ---------------------------------------------------------------------
# 3. Streamlit Interface
# ---------------------------------------------------------------------
def sarimax_interface(df):

    st.header("🔮 SARIMAX Forecasting of Energy Production / Consumption")

    df["startTime"] = pd.to_datetime(df["startTime"]).dt.tz_localize(None)
    df = df.sort_values("startTime")

    st.subheader("Select Data")

    price_area = st.selectbox("Price Area:", sorted(df["priceArea"].unique()))
    production_group = st.selectbox("Production Group:", sorted(df["productionGroup"].unique()))

    df_filtered = df[(df["priceArea"] == price_area) & (df["productionGroup"] == production_group)].copy()

    if df_filtered.empty:
        st.error("No data for this selection.")
        return

    # Select target property
    target_col = st.selectbox(
        "Target Forecast Property:",
        ["quantityKwh"]
    )

    # Select exogenous variables (optional)
    numeric_cols = ["quantityKwh"]
    exog_choices = st.multiselect(
        "Select Exogenous Variables (optional):",
        [c for c in df_filtered.columns if c not in ["startTime", "priceArea", "productionGroup"]],
        default=[]
    )

    st.subheader("Training Timeframe")

    min_date = df_filtered["startTime"].min().to_pydatetime()
    max_date = df_filtered["startTime"].max().to_pydatetime()

    train_range = st.slider(
        "Select training period:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date - pd.Timedelta(days=60)),
        format="YYYY-MM-DD"
    )
    train_start, train_end = train_range

    forecast_steps = st.number_input("Forecast Horizon (hours):", 1, 1000, 48)

    st.subheader("SARIMAX Parameters")

    p = st.number_input("p (AR):", 0, 10, 1)
    d = st.number_input("d (Diff):", 0, 2, 1)
    q = st.number_input("q (MA):", 0, 10, 1)

    P = st.number_input("P (Seasonal AR):", 0, 10, 1)
    D = st.number_input("D (Seasonal Diff):", 0, 2, 1)
    Q = st.number_input("Q (Seasonal MA):", 0, 10, 1)
    s = st.number_input("Seasonal period (s):", 1, 500, 24)

    trend = st.selectbox("Trend:", ["n", "c", "t", "ct"])

    order = (p, d, q)
    seasonal_order = (P, D, Q, s)

    if st.button("Run SARIMAX Forecast"):
        with st.spinner("Running SARIMAX..."):
            results, forecast_mean, forecast_ci = run_sarimax_forecast(
                df_filtered,
                target_col,
                exog_choices,
                train_start,
                train_end,
                forecast_steps,
                order,
                seasonal_order,
                trend,
            )

        y_full = df_filtered.set_index("startTime")[target_col]

        fig = plot_sarimax_forecast(
            y=y_full,
            forecast_mean=forecast_mean,
            forecast_ci=forecast_ci,
            train_end=train_end,
            title=f"SARIMAX Forecast — {price_area} ({production_group})"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Summary")
        st.text(results.summary())


# ---------------------------------------------------------------------
# 4. MongoDB Loaders
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_mongo_collection():
    uri = st.secrets.get("MONGO_URI")
    client = MongoClient(uri, tlsAllowInvalidCertificates=True)
    return client["assignment4"]["elhub"]


@st.cache_data(show_spinner=False)
def load_energy(year: int) -> pd.DataFrame:

    coll = get_mongo_collection()

    cursor = coll.find(
        {
            "startTime": {
                "$gte": datetime(year, 1, 1),
                "$lt": datetime(year + 1, 1, 1),
            }
        },
        {"_id": 0}
    )

    df = pd.DataFrame(list(cursor))
    df["startTime"] = pd.to_datetime(df["startTime"]).dt.tz_localize(None)
    df = df.dropna(subset=["startTime"]).sort_values("startTime")

    return df


# ---------------------------------------------------------------------
# Run App
# ---------------------------------------------------------------------
df = load_energy(st.session_state.year)
sarimax_interface(df)
