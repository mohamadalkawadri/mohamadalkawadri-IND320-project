import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import streamlit as st
import plotly.graph_objects as go
from typing import Optional

def mask_month_range(df: pd.DataFrame, start_month, end_month) -> pd.Series:
    return (df["time"].dt.month >= start_month) & (df["time"].dt.month <= end_month)

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def download_era5_hourly(latitude: float, longitude: float, year: int) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m"
        ],
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
        "wind_gusts_10m": hourly.Variables(3).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(4).ValuesAsNumpy(),
    }

    df = pd.DataFrame(data)
    return df

def plot_stl_decomposition(
    df: pd.DataFrame,
    area: str = "NO5",
    group: str = "Hydro",
    period: int = 24*7,            # default: one week of hourly seasonality
    seasonal_smoother: int = 13,   # smoother window for seasonal component
    trend_smoother: int = 301,     # smoother window for trend component
    robust: bool = True            # robust to outliers
):
    """
    Perform STL decomposition (Seasonal-Trend decomposition using LOESS)
    on electricity production data filtered by area and production group,
    and display it dynamically in Streamlit using Plotly.
    """

    # --- 1. Filter data for chosen area & group ---
    mask = (df["priceArea"] == area) & (df["productionGroup"] == group)
    data = df.loc[mask].copy()

    if data.empty:
        st.warning(f"No data found for area '{area}' and group '{group}'.")
        return

    data = data.sort_values("startTime")
    y = data["quantityKwh"].astype(float).values
    t = pd.to_datetime(data["startTime"])

    # --- 2. Apply STL decomposition ---
    stl = STL(
        y,
        period=period,
        seasonal=seasonal_smoother,
        trend=trend_smoother,
        robust=robust
    ).fit()

    # --- 3. Create interactive Plotly figure ---
    fig = make_stl_plotly(t, y, stl, area, group)

    # --- 4. Display in Streamlit ---
    st.plotly_chart(fig, use_container_width=True)


def make_stl_plotly(t, y, stl, area, group):
    """Helper function to construct a 4-row interactive STL Plot."""
    import plotly.subplots as sp

    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=["Original", "Trend", "Seasonal", "Residual"])

    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Original', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=stl.trend, mode='lines', name='Trend', line=dict(color='darkorange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=stl.seasonal, mode='lines', name='Seasonal', line=dict(color='seagreen')), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=stl.resid, mode='lines', name='Residual', line=dict(color='gray')), row=4, col=1)

    fig.update_layout(
        height=900,
        title=f"STL Decomposition — {area} | {group}",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(title_text="Time", row=4, col=1)
    return fig

def plot_production_spectrogram(
    df: pd.DataFrame,
    area: str = "NO5",
    group: str = "Hydro",
    window_length: int = 256,
    overlap: int = 128
):
    """
    Compute and plot an interactive spectrogram for Elhub production data
    using Plotly in Streamlit.
    """

    # --- 1. Filter relevant subset ---
    mask = (df["priceArea"] == area) & (df["productionGroup"] == group)
    data = df.loc[mask].copy()

    if data.empty:
        st.warning(f"No data found for area '{area}' and group '{group}'.")
        return

    data = data.sort_values("startTime")
    y = data["quantityKwh"].astype(float).values
    t = pd.to_datetime(data["startTime"])

    # --- 2. Compute spectrogram ---
    fs = 1.0  # sampling frequency (1/hour if hourly data)
    f, t_spec, Sxx = spectrogram(
        y,
        fs=fs,
        nperseg=window_length,
        noverlap=overlap,
        scaling="density",
        detrend="linear",
        mode="magnitude"
    )

    # Convert to decibel scale for better visualization
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # --- 3. Build interactive Plotly figure ---
    fig = go.Figure(data=go.Heatmap(
        x=t_spec,
        y=f,
        z=Sxx_db,
        colorscale="Viridis",
        colorbar=dict(title="Power Spectral Density (dB)"),
        hovertemplate="Time index: %{x}<br>Frequency: %{y:.2f} 1/h<br>Power: %{z:.2f} dB<extra></extra>",
    ))

    fig.update_layout(
        title=f"Spectrogram — {area} | {group}",
        xaxis_title="Time (index)",
        yaxis_title="Frequency [1/hour]",
        height=700,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    # --- 4. Display in Streamlit ---
    st.plotly_chart(fig, use_container_width=True)

from scipy.fftpack import dct, idct

def compute_satv_spc_outliers(
    df: pd.DataFrame,
    cutoff: int = 100,
    n_sigma: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute SATV (Seasonally Adjusted Temperature Variation) using DCT
    and detect SPC-based outliers on SATV.

    This function fulfills the assignment requirement:

    - Perform a DCT of the temperature time series
    - High-pass filter by zeroing low-frequency components (cutoff)
    - Inverse DCT → SATV(DCT)
    - Compute SPC limits on SATV using a robust std (MAD)
    - Mark points outside the SPC band as outliers

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'temperature_2m' and 'time'.
    cutoff : int
        Number of low-frequency DCT coefficients to zero out
        (removes seasonality / low-frequency trend).
    n_sigma : float
        Number of robust standard deviations (sigma) for SPC limits.

    Returns
    -------
    result_df : pd.DataFrame
        Copy of df with added columns:
        - 'SATV': seasonally adjusted temperature variation
        - 'is_outlier': boolean mask of SPC outliers
    stats : dict
        Dictionary with SPC stats (median, robust_std, limits, etc.).
    """
    if "temperature_2m" not in df.columns or "time" not in df.columns:
        raise ValueError("DataFrame must contain 'temperature_2m' and 'time' columns.")

    # Sort by time to ensure correct ordering
    work = df.sort_values("time").reset_index(drop=True).copy()

    temp = work["temperature_2m"].astype(float).to_numpy()

    if len(temp) <= cutoff:
        raise ValueError(
            f"cutoff={cutoff} is too large for series of length {len(temp)}. "
            "Use a smaller cutoff."
        )

    # --- 1. DCT and high-pass filter ---
    dct_coeffs = dct(temp, norm="ortho")
    dct_high = np.copy(dct_coeffs)
    dct_high[:cutoff] = 0  # remove low frequencies
    satv = idct(dct_high, norm="ortho")  # SATV(DCT)

    # --- 2. Robust SPC limits (on SATV) using MAD ---
    median_satv = float(np.median(satv))
    mad = float(np.median(np.abs(satv - median_satv)))
    # Robust std from MAD (if MAD is zero, fall back to classical std)
    if mad > 0:
        robust_std = 1.4826 * mad
    else:
        robust_std = float(np.std(satv, ddof=1))

    upper_limit = median_satv + n_sigma * robust_std
    lower_limit = median_satv - n_sigma * robust_std

    # --- 3. Outlier mask on SATV(DCT) ---
    outlier_mask = (satv > upper_limit) | (satv < lower_limit)

    # Attach to DataFrame
    work["SATV"] = satv
    work["is_outlier"] = outlier_mask

    stats = {
        "n_observations": len(work),
        "n_outliers": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()),
        "median_satv": median_satv,
        "mad": mad,
        "robust_std": float(robust_std),
        "upper_limit": float(upper_limit),
        "lower_limit": float(lower_limit),
        "cutoff": cutoff,
        "n_sigma": float(n_sigma),
    }

    return work, stats


from plotly.subplots import make_subplots

def plot_temperature_with_spc(
    df: pd.DataFrame,
    cutoff: int = 100,
    n_sigma: float = 3.0,
    city: str = "Unknown"
):
    """
    Interactive SPC plot for temperature data based on DCT-derived SATV.

    This uses `compute_satv_spc_outliers` to:
    - perform the DCT,
    - compute SATV(DCT),
    - set SPC limits,
    - mark outliers.

    The figure shows:
    - Temperature vs time (with outliers marked),
    - SATV(DCT) vs time with SPC limits.
    """

    # --- 1. Compute SATV + SPC + outliers (assignment requirement) ---
    try:
        work, stats = compute_satv_spc_outliers(
            df,
            cutoff=cutoff,
            n_sigma=n_sigma,
        )
    except ValueError as e:
        st.error(str(e))
        return

    time = pd.to_datetime(work["time"])
    temp = work["temperature_2m"].astype(float).to_numpy()
    satv = work["SATV"].to_numpy()
    outlier_mask = work["is_outlier"].to_numpy()

    upper_limit = stats["upper_limit"]
    lower_limit = stats["lower_limit"]
    robust_std = stats["robust_std"]

    # --- 2. Build interactive two-panel Plotly figure ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Temperature (°C) with outliers",
            "SATV (DCT high-pass) with SPC limits",
        ),
    )

    # Panel 1: Temperature
    fig.add_trace(
        go.Scatter(
            x=time, y=temp,
            mode="lines",
            name="Temperature (°C)",
            line=dict(color="steelblue"),
            hovertemplate="Time: %{x}<br>Temp: %{y:.2f} °C<extra></extra>",
        ),
        row=1, col=1,
    )

    # Highlight outliers on temperature
    fig.add_trace(
        go.Scatter(
            x=time[outlier_mask],
            y=temp[outlier_mask],
            mode="markers",
            name="Outliers (from SATV SPC)",
            marker=dict(color="crimson", size=6, line=dict(width=0.5, color="white")),
            hovertemplate="⛔ Outlier<br>Time: %{x}<br>Temp: %{y:.2f} °C<extra></extra>",
        ),
        row=1, col=1,
    )

    # Panel 2: SATV with SPC limits
    fig.add_trace(
        go.Scatter(
            x=time, y=satv,
            mode="lines",
            name="SATV (DCT)",
            line=dict(color="slategray"),
            hovertemplate="Time: %{x}<br>SATV: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # SPC upper & lower limits on SATV
    fig.add_trace(
        go.Scatter(
            x=time, y=[upper_limit] * len(time),
            mode="lines",
            name=f"Upper SPC limit (+{n_sigma}σ)",
            line=dict(color="orange", dash="dash"),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=[lower_limit] * len(time),
            mode="lines",
            name=f"Lower SPC limit (-{n_sigma}σ)",
            line=dict(color="orange", dash="dash"),
        ),
        row=2, col=1,
    )

    # --- 3. Layout ---
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="SATV", row=2, col=1)

    fig.update_layout(
        title=f"SPC on SATV(DCT) — {city}",
        height=700,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # --- 4. Summary for Streamlit ---
    summary = {
        "city": city,
        "n_observations": stats["n_observations"],
        "n_outliers": stats["n_outliers"],
        "outlier_fraction (%)": round(100 * stats["outlier_fraction"], 2),
        "robust_std": round(robust_std, 3),
        "upper_limit": round(upper_limit, 3),
        "lower_limit": round(lower_limit, 3),
        "cutoff": cutoff,
        "n_sigma": n_sigma,
    }

    # Display
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Summary statistics")
    st.json(summary)

    outliers = work[work["is_outlier"]].copy()
    if not outliers.empty:
        with st.expander("Show outlier data (SATV-based SPC)"):
            st.dataframe(outliers[["time", "temperature_2m", "SATV", "is_outlier"]])
    else:
        st.info("No outliers detected according to SATV-based SPC limits.")

    return fig, summary, outliers


def plot_precipitation_with_lof(
    df: pd.DataFrame,
    contamination: float = 0.01,  # proportion of outliers
    city: str = "Unknown"
):
    """
    Interactive LOF-based precipitation anomaly detection using Plotly and Streamlit.
    """

    # --- 1. Validate data ---
    if "precipitation" not in df.columns or "time" not in df.columns:
        st.error("DataFrame must contain columns: 'precipitation' and 'time'.")
        return

    if len(df) < 10:
        st.warning("Not enough data points to compute LOF reliably.")
        return

    # --- 2. Extract data ---
    precip = df["precipitation"].astype(float).values.reshape(-1, 1)
    time = pd.to_datetime(df["time"])

    # --- 3. Fit LOF model ---
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    labels = lof.fit_predict(precip)  # -1 = outlier, 1 = inlier
    lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous

    # --- 4. Identify outliers ---
    outlier_mask = labels == -1
    outliers = df[outlier_mask].copy()
    outliers["lof_score"] = lof_scores[outlier_mask]

    # --- 5. Build interactive Plotly figure ---
    fig = go.Figure()

    # Normal precipitation
    fig.add_trace(go.Scatter(
        x=time[~outlier_mask],
        y=precip[~outlier_mask].flatten(),
        mode="lines+markers",
        name="Precipitation (mm)",
        marker=dict(color="dodgerblue", size=5),
        line=dict(width=1, color="dodgerblue"),
        hovertemplate="Time: %{x}<br>Precipitation: %{y:.2f} mm<extra></extra>"
    ))

    # Outliers
    fig.add_trace(go.Scatter(
        x=time[outlier_mask],
        y=precip[outlier_mask].flatten(),
        mode="markers",
        name="Anomalies",
        marker=dict(color="crimson", size=7, line=dict(color="white", width=0.5)),
        hovertemplate="⛔ Anomaly<br>Time: %{x}<br>Precipitation: %{y:.2f} mm<extra></extra>"
    ))

    # --- 6. Layout customization ---
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Precipitation (mm)",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- 7. Summary statistics ---
    summary = {
        "city": city,
        "n_observations": len(df),
        "n_outliers": int(outlier_mask.sum()),
        "outlier_fraction (%)": round(100 * outlier_mask.mean(), 2),
        "mean_precip (mm)": round(np.mean(precip), 3),
        "max_precip (mm)": round(np.max(precip), 3),
        "median_outlier_score": (
            round(np.median(outliers["lof_score"]), 3) if not outliers.empty else None
        ),
    }

    # --- 8. Display in Streamlit ---
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Summary statistics")
    st.json(summary)

    if not outliers.empty:
        with st.expander("Show detected anomalies"):
            st.dataframe(outliers[["time", "precipitation", "lof_score"]])
    else:
        st.info("No significant anomalies detected.")

    return fig, summary, outliers


def compute_sliding_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    window: int,
    lag: int = 0,
    min_periods: Optional[int] = None,
):
    """
    Compute a rolling Pearson correlation between two columns with an optional lag.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing x_col and y_col with a datetime index (or a 'time' column).
    x_col, y_col : str
        Column names for the two series to correlate.
    window : int
        Rolling window size in number of samples (e.g., hours).
    lag : int
        Lead/lag applied to y_col before computing the correlation.
        Positive lag shifts y_col forward in time (correlate x_t with y_{t+lag}).
    min_periods : int | None
        Minimum number of observations needed in the window; defaults to half the window.

    Returns
    -------
    pd.DataFrame
        Copy of df with an added 'rolling_corr' column aligned to df's index.
    float
        Overall Pearson correlation between x_col and the lagged y_col.
    """
    if min_periods is None:
        min_periods = max(5, window // 2)

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must exist in the DataFrame.")

    work = df[[x_col, y_col]].copy()
    work[y_col] = work[y_col].shift(lag)
    work["rolling_corr"] = (
        work[x_col]
        .rolling(window=window, min_periods=min_periods)
        .corr(work[y_col])
    )

    overall_corr = work[x_col].corr(work[y_col])

    result = df.copy()
    result["rolling_corr"] = work["rolling_corr"]
    return result, overall_corr
