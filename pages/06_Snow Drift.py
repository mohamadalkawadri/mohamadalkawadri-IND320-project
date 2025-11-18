import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import download_era5_hourly
from sidebar import setup_sidebar

setup_sidebar()

# -----------------------------
#  Snow drift helper functions
# -----------------------------

def compute_Qupot(hourly_wind_speeds, dt=3600):
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total

def sector_index(direction):
    return int(((direction + 11.25) % 360) // 22.5)

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe

    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"

    Qt = Qinf * (1 - 0.14 ** (F / T))

    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control,
    }


# -----------------------------------
#  Seasonal (July–June) Snow Transport
# -----------------------------------
def compute_yearly_results(df, T, F, theta):
    seasons = sorted(df['season'].unique())
    results_list = []

    for s in seasons:
        season_start = pd.Timestamp(year=s, month=7, day=1)
        season_end   = pd.Timestamp(year=s+1, month=6, day=30, hour=23, minute=59)

        df_season = df[(df['time'] >= season_start) & (df['time'] <= season_end)]
        if df_season.empty:
            continue

        df_season = df_season.copy()
        df_season['Swe_hourly'] = df_season.apply(
            lambda r: r['precipitation'] if r['temperature_2m'] < 1 else 0,
            axis=1
        )
        total_Swe = df_season['Swe_hourly'].sum()
        wind_speeds = df_season["wind_speed_10m"].tolist()

        result = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
        result["season_label"] = f"{s}-{s+1}"
        result["season_start_year"] = s
        results_list.append(result)

    return pd.DataFrame(results_list)


# -----------------------------------
#      Monthly Snow Drift Qt
# -----------------------------------
def compute_monthly_results(df, T, F, theta):
    df = df.copy()

    df["Swe_hourly"] = df.apply(
        lambda r: r["precipitation"] if r["temperature_2m"] < 1 else 0,
        axis=1
    )

    groups = df.groupby([df["time"].dt.year, df["time"].dt.month])

    records = []
    for (year, month), group in groups:
        if group.empty:
            continue

        total_Swe = group["Swe_hourly"].sum()
        wind_speeds = group["wind_speed_10m"].tolist()

        res = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
        records.append({
            "year": year,
            "month": month,
            "year_month": f"{year}-{month:02d}",
            "Qt (kg/m)": res["Qt (kg/m)"],
        })

    return pd.DataFrame(records)


# -----------------------------------
#        Wind Rose Plot
# -----------------------------------
def compute_average_sector(df):
    sectors_list = []

    for s, g in df.groupby('season'):
        g = g.copy()
        g['Swe_hourly'] = g.apply(
            lambda r: r['precipitation'] if r['temperature_2m'] < 1 else 0,
            axis=1
        )
        ws = g["wind_speed_10m"].tolist()
        wdir = g["wind_direction_10m"].tolist()
        sectors_list.append(compute_sector_transport(ws, wdir))

    if not sectors_list:
        return np.zeros(16)

    return np.mean(sectors_list, axis=0)

def plot_rose(avg_sector_values, overall_avg):
    avg_tonnes = np.array(avg_sector_values) / 1000.0

    directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                  'S','SSW','SW','WSW','W','WNW','NW','NNW']
    angles = np.arange(0,360,360/16)

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=avg_tonnes,
        theta=angles,
        width=[360/16]*16,
        marker=dict(line=dict(color="black", width=1)),
        hovertemplate="<b>%{theta}°</b><br>Qt: %{r:.2f} t/m<extra></extra>"
    ))

    fig.update_layout(
        title=f"Average directional distribution<br>Qt = {overall_avg/1000:.1f} tonnes/m",
        polar=dict(
            angularaxis=dict(
                tickmode="array", tickvals=angles, ticktext=directions,
                direction="clockwise", rotation=90
            )
        ),
        showlegend=False
    )
    return fig


# -----------------------------------
#   Load Open-Meteo Data Combined
# -----------------------------------
@st.cache_data(show_spinner=False)
def load_meteo_for_coords(lat, lon, start_year=2021, end_year=2024):
    dfs = []
    for year in range(start_year, end_year+1):
        dfs.append(download_era5_hourly(lat, lon, year))

    df = pd.concat(dfs, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["season"] = df["time"].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    return df


# ---------------------------------------------------------
#                  STREAMLIT PAGE
# ---------------------------------------------------------
st.title("❄️ Snow Drift Calculation & Wind Rose")

# Coordinates from map page
coords = st.session_state.get("selected_coord", None)

if coords is None:
    st.warning("Please go to the map page and choose a location.")
    st.stop()

lat = coords["lat"]
lon = coords["lon"]

st.markdown(f"**Using coordinates:** `{lat:.4f}`, `{lon:.4f}`")

start_year, end_year = st.slider(
    "Select year range (season start year):",
    min_value=2021, max_value=2024,
    value=(2021, 2024), step=1
)

# Load meteo
df = load_meteo_for_coords(lat, lon)

T = 3000
F = 30000
theta = 0.5

# ----------------------------
#  Yearly (Seasonal) Results
# ----------------------------
yearly_df = compute_yearly_results(df, T, F, theta)
yearly_df = yearly_df[
    (yearly_df["season_start_year"] >= start_year) &
    (yearly_df["season_start_year"] <= end_year)
].copy()

yearly_df["Qt (tonnes/m)"] = yearly_df["Qt (kg/m)"] / 1000
yearly_df["date"] = yearly_df["season_start_year"].apply(lambda y: pd.Timestamp(year=y+1, month=1, day=1))


# Plot yearly Qt
fig_year = go.Figure(go.Bar(
    x=yearly_df["season_label"],
    y=yearly_df["Qt (tonnes/m)"],
    text=[f"{v:.1f}" for v in yearly_df["Qt (tonnes/m)"]],
    textposition="auto",
    marker_color="steelblue",
))
fig_year.update_layout(
    title="Yearly Snow Drift Qt (Season July–June)",
    xaxis_title="Season",
    yaxis_title="Qt (tonnes/m)",
    template="plotly_white"
)
fig_year.update_xaxes(tickangle=-45)
st.plotly_chart(fig_year, use_container_width=True)

# ----------------------------
#  Monthly Results
# ----------------------------
monthly_df = compute_monthly_results(df, T, F, theta)
monthly_df = monthly_df[
    (monthly_df["year"] >= start_year) &
    (monthly_df["year"] <= end_year)
].copy()
monthly_df["Qt (tonnes/m)"] = monthly_df["Qt (kg/m)"] / 1000
monthly_df["date"] = pd.to_datetime(monthly_df["year_month"] + "-01")

# Plot monthly + yearly combined
st.subheader("📅 Monthly Snow Drift vs Yearly Snow Drift")
fig_month = go.Figure()

# Monthly Qt
fig_month.add_trace(go.Scatter(
    x=monthly_df["date"],
    y=monthly_df["Qt (tonnes/m)"],
    mode="lines+markers",
    name="Monthly Qt",
    line=dict(color="orange"),
    hovertemplate="Month: %{x|%Y-%m}<br>Qt: %{y:.2f} t/m<extra></extra>"
))

# Yearly Qt
fig_month.add_trace(go.Scatter(
    x=yearly_df["date"],
    y=yearly_df["Qt (tonnes/m)"],
    mode="markers+text",
    text=[f"{v:.1f}" for v in yearly_df["Qt (tonnes/m)"]],
    textposition="top center",
    marker=dict(color="red", size=12),
    name="Yearly Qt",
    hovertemplate="Season: %{x|%Y}<br>Qt: %{y:.2f} t/m<extra></extra>"
))

fig_month.update_layout(
    title="Monthly and Yearly Snow Drift (Qt)",
    xaxis_title="Time",
    yaxis_title="Qt (tonnes/m)",
    template="plotly_white",
    height=600
)
fig_month.update_xaxes(tickformat="%Y-%m", tickangle=-45)
st.plotly_chart(fig_month, use_container_width=True)


# ----------------------------
#  Wind Rose
# ----------------------------
df_range = df[df["season"].isin(yearly_df["season_start_year"])]
avg_sectors = compute_average_sector(df_range)
overall_avg = yearly_df["Qt (kg/m)"].mean()

st.subheader("🌬️ Wind Rose")
fig_rose = plot_rose(avg_sectors, overall_avg)
st.plotly_chart(fig_rose, use_container_width=True)
