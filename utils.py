import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

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
    on electricity production data filtered by area and production group.
    """

    # --- 1. Filter data for chosen area & group ---
    mask = (df["priceArea"] == area) & (df["productionGroup"] == group)
    data = df.loc[mask].copy()

    if data.empty:
        raise ValueError(f"No data found for area={area} and group={group}")

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

    # --- 3. Plot decomposition ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"STL Decomposition — {area} | {group}", fontsize=14)

    axes[0].plot(t, y, color="steelblue")
    axes[0].set_ylabel("Original")

    axes[1].plot(t, stl.trend, color="darkorange")
    axes[1].set_ylabel("Trend")

    axes[2].plot(t, stl.seasonal, color="seagreen")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(t, stl.resid, color="gray")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Time")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return plt


def plot_production_spectrogram(
    df: pd.DataFrame,
    area: str = "NO5",
    group: str = "Hydro",
    window_length: int = 256,   # number of samples per FFT window
    overlap: int = 128          # overlap between windows
):
    """
    Compute and plot a spectrogram for Elhub production data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Elhub production data with columns: priceArea, productionGroup, startTime, quantityKwh
    area : str
        Electricity price area (e.g., "NO1"..."NO5")
    group : str
        Production group (e.g., "Hydro", "Wind", "Thermal")
    window_length : int
        FFT window length (number of samples per segment)
    overlap : int
        Number of overlapping samples between windows
    """
    
    # --- 1. Filter relevant subset ---
    mask = (df["priceArea"] == area) & (df["productionGroup"] == group)
    data = df.loc[mask].copy()
    if data.empty:
        raise ValueError(f"No data found for area={area} and group={group}")
        
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

    # --- 3. Plot ---
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
    plt.colorbar(label="Power Spectral Density (dB)")
    plt.title(f"Spectrogram — {area} | {group}")
    plt.ylabel("Frequency [1/hour]")
    plt.xlabel("Time [index]")
    plt.tight_layout()
    plt.show()
    
    return plt

from scipy.fftpack import dct, idct

def plot_temperature_with_spc(
    df: pd.DataFrame,
    cutoff: int = 100,          # DCT high-pass cutoff frequency
    n_sigma: float = 3.0,       # number of robust stds (MADs) for control limits
    city: str = "Unknown"
):
    """
    Plot temperature vs time, perform DCT high-pass filtering to compute
    seasonally adjusted temperature variations (SATV), and visualize SPC boundaries.
    """
    
    # --- 1. Prepare temperature data ---
    temp = df["temperature_2m"].values
    time = pd.to_datetime(df["time"])
    
    # --- 2. Apply Discrete Cosine Transform (DCT) for high-pass filtering ---
    dct_coeffs = dct(temp, norm='ortho')
    
    # Remove low-frequency (seasonal) components: keep only high frequencies
    dct_high = np.copy(dct_coeffs)
    dct_high[:cutoff] = 0  # zero out low frequencies
    
    satv = idct(dct_high, norm='ortho')  # seasonally adjusted temp variations (HP-filtered)
    
    # --- 3. Compute robust SPC limits using MAD ---
    median_satv = np.median(satv)
    mad = np.median(np.abs(satv - median_satv))
    robust_std = 1.4826 * mad  # convert MAD to std estimate
    
    upper_limit = median_satv + n_sigma * robust_std
    lower_limit = median_satv - n_sigma * robust_std
    
    # --- 4. Identify outliers ---
    outlier_mask = (satv > upper_limit) | (satv < lower_limit)
    outliers = df[outlier_mask].copy()
    outliers["SATV"] = satv[outlier_mask]
    
    # --- 5. Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(time, temp, label="Temperature (°C)", color="steelblue", linewidth=1)
    plt.scatter(time[outlier_mask], temp[outlier_mask], color="crimson", label="Outliers", s=20)
    
    plt.axhline(np.median(temp), color="gray", linestyle="--", linewidth=1, label="Median")
    plt.title(f"Temperature and SPC Outlier Detection — {city}")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- 6. Summary ---
    summary = {
        "city": city,
        "n_observations": len(df),
        "n_outliers": outlier_mask.sum(),
        "outlier_fraction (%)": round(100 * outlier_mask.mean(), 2),
        "robust_std": robust_std,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit
    }
    
    return plt, summary, outliers


def plot_precipitation_with_lof(
    df: pd.DataFrame,
    contamination: float = 0.01,  # proportion of outliers (default = 1%)
    city: str = "Unknown"
):
    """
    Plot precipitation vs time and detect anomalies using Local Outlier Factor (LOF).
    """
    # --- 1. Extract relevant series ---
    precip = df["precipitation"].values.reshape(-1, 1)
    time = pd.to_datetime(df["time"])

    # --- 2. Fit LOF model ---
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    labels = lof.fit_predict(precip)  # -1 = outlier, 1 = inlier
    lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous

    # --- 3. Identify outliers ---
    outlier_mask = labels == -1
    outliers = df[outlier_mask].copy()
    outliers["lof_score"] = lof_scores[outlier_mask]

    # --- 4. Plot results ---
    plt.figure(figsize=(12, 5))
    plt.plot(time, precip, label="Precipitation (mm)", color="dodgerblue", linewidth=1)
    plt.scatter(time[outlier_mask], precip[outlier_mask], color="crimson", label="Anomalies", s=20)
    
    plt.title(f"Precipitation Anomaly Detection (LOF) — {city}")
    plt.xlabel("Time")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 5. Summary statistics ---
    summary = {
        "city": city,
        "n_observations": len(df),
        "n_outliers": outlier_mask.sum(),
        "outlier_fraction (%)": round(100 * outlier_mask.mean(), 2),
        "mean_precip (mm)": np.mean(precip),
        "max_precip (mm)": np.max(precip),
        "median_outlier_score": np.median(outliers["lof_score"]) if not outliers.empty else None
    }

    return plt, summary, outliers