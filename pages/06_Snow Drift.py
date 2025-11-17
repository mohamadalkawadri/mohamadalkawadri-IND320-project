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
    """
    Potential wind-driven snow transport (Qupot) [kg/m],
    summed hourly using u^3.8.
    """
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total

def sector_index(direction):
    """
    Map wind direction (deg) to index 0–15 in a 16-sector rose.
    """
    return int(((direction + 11.25) % 360) // 22.5)

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """
    Cumulative transport per 16 sectors (kg/m).
    """
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """ 
    Tabler (2003) snow drift transport.
    Returns dict with Qupot, Qspot, Srwe, Qinf, Qt, Control.
    """
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe      # snowfall-limited [kg/m]
    Srwe = theta * Swe         # relocated water equivalent [mm]

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

def compute_yearly_results(df, T, F, theta):
    """
    Compute seasonal snow transport per 'year', where a year is:
    1 July N – 30 June N+1.

    Expects:
      df['time'] as datetime64
      df['season'] already defined: season = year if month>=7 else year-1
      df has columns:
        - 'precipitation (mm)'
        - 'temperature_2m (°C)'
        - 'wind_speed_10m (m/s)'
    """
    seasons = sorted(df['season'].unique())
    results_list = []

    for s in seasons:
        season_start = pd.Timestamp(year=s, month=7, day=1)
        season_end = pd.Timestamp(year=s + 1, month=6, day=30, hour=23, minute=59, second=59)

        df_season = df[(df['time'] >= season_start) & (df['time'] <= season_end)]
        if df_season.empty:
            continue

        df_season = df_season.copy()
        df_season['Swe_hourly'] = df_season.apply(
            lambda row: row['precipitation'] if row['temperature_2m'] < 1 else 0,
            axis=1
        )
        total_Swe = df_season['Swe_hourly'].sum()
        wind_speeds = df_season["wind_speed_10m"].tolist()

        result = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
        result["season_label"] = f"{s}-{s+1}"    # e.g. "2021-2022"
        result["season_start_year"] = s          # numeric start year
        results_list.append(result)

    return pd.DataFrame(results_list)

def compute_average_sector(df):
    """
    Average sector-wise transport over the seasons present in df.
    df must contain:
      - 'season'
      - 'precipitation (mm)'
      - 'temperature_2m (°C)'
      - 'wind_speed_10m (m/s)'
      - 'wind_direction_10m (°)'
    """
    sectors_list = []

    for s, group in df.groupby('season'):
        group = group.copy()
        group['Swe_hourly'] = group.apply(
            lambda row: row['precipitation'] if row['temperature_2m'] < 1 else 0,
            axis=1
        )
        ws = group["wind_speed_10m"].tolist()
        wdir = group["wind_direction_10m"].tolist()
        sectors = compute_sector_transport(ws, wdir)
        sectors_list.append(sectors)

    if not sectors_list:
        return np.zeros(16)

    avg_sectors = np.mean(sectors_list, axis=0)
    return avg_sectors

def plot_rose(avg_sector_values, overall_avg):
    """
    Interactive Plotly wind rose — 16 sectors.
    avg_sector_values: list of 16 values (kg/m)
    overall_avg: scalar (kg/m)
    """

    # Convert to tonnes
    avg_tonnes = np.array(avg_sector_values) / 1000.0

    # Directions (16-sector rose)
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Full circle angles
    angles_deg = np.arange(0, 360, 360 / 16)

    # Plotly barpolar
    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=avg_tonnes,
        theta=angles_deg,
        width=[360/16]*16,
        marker=dict(
            line=dict(color="black", width=1)
        ),
        hovertemplate="<b>%{theta}°</b><br>" +
                      "Qt: %{r:.2f} tonnes/m<extra></extra>"
    ))

    # Average Qt text
    overall_tonnes = overall_avg / 1000.0

    fig.update_layout(
        title=f"Average directional distribution<br>Qt = {overall_tonnes:,.1f} tonnes/m",
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=angles_deg,
                ticktext=directions,
                direction="clockwise",
                rotation=90  # zero at North
            ),
            radialaxis=dict(
                ticksuffix=" t/m"
            )
        ),
        showlegend=False,
    )

    return fig

# -------------------------------------
#  Data loading for a given coordinate
# -------------------------------------

@st.cache_data(show_spinner=False)
def load_meteo_for_coords(lat, lon, start_year=2021, end_year=2024):
    """
    Load hourly met data for given coords.

    For now this just reads a CSV like in Snow_drift.py.
    Adapt this to your Open-Meteo API or filename scheme.
    """
    # EXAMPLE: fixed file (from your original script)

    dfs = []
    for year in range(start_year, end_year + 1):
        dfs.append(download_era5_hourly(lat, lon, year))

    df = pd.concat(dfs, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize(None)

    # Define season: July–June (same logic as in original script)
    df["season"] = df["time"].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
    return df

# ---------------
#  Streamlit page
# ---------------
st.title("Snow drift calculation and wind rose")

# 1) Check that coordinates exist from map page
#    (adapt the key to whatever you actually use there)
coord_key_candidates = ["selected_coord"]

coords = None
if "selected_coord" in st.session_state:
    coords = st.session_state["selected_coord"]

if coords is None:
    st.warning(
        "No coordinates selected.\n\n"
        "Please go to the map page and choose a location first."
    )
    st.stop()

lat = coords.get("lat")
lon = coords.get("lon")

st.markdown(f"**Using coordinates:** lat = `{lat:.4f}`, lon = `{lon:.4f}`")


start_year, end_year = st.slider(
    "Select year range (season start year):",
    min_value=2021,
    max_value=2024,
    value=(2021, 2024),
    step=1,
)

# 2) Load met data for those coordinates
df = load_meteo_for_coords(lat, lon)

if df.empty:
    st.error("No meteorological data available for these coordinates.")
    st.stop()

# Parameters (you can expose these as widgets if you want)
T = 3000      # max transport distance (m)
F = 30000     # fetch distance (m)
theta = 0.5   # relocation coefficient

# 3) Compute yearly snow drift for all seasons
yearly_df = compute_yearly_results(df, T, F, theta)
if yearly_df.empty:
    st.error("No valid seasonal snow drift data found in the dataset.")
    st.stop()


mask_range = (yearly_df["season_start_year"] >= start_year) & \
                (yearly_df["season_start_year"] <= end_year)
yearly_range = yearly_df[mask_range].copy()

if yearly_range.empty:
    st.warning("No seasons in the selected year range.")
    st.stop()

# 5) Plot snow drift per year (Qt in tonnes/m)
yearly_range["Qt (tonnes/m)"] = yearly_range["Qt (kg/m)"] / 1000.0

fig1 = go.Figure(go.Bar(
    x=yearly_range["season_label"],
    y=yearly_range["Qt (tonnes/m)"],
    text=[f"{v:,.1f}" for v in yearly_range["Qt (tonnes/m)"]],
    textposition="auto",
    marker_color="steelblue",
    hovertemplate="Season: %{x}<br>Qt: %{y:.2f} tonnes/m<extra></extra>",
))
fig1.update_layout(
    xaxis_title="Season (July–June)",
    yaxis_title="Qt (tonnes/m)",
    title="Snow drift per year for selected range",
    margin=dict(l=40, r=40, t=60, b=60),
    template="plotly_white",
)
fig1.update_xaxes(tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)

# 6) Wind rose for the same year range
#    Filter original df to only those seasons
seasons_selected = yearly_range["season_start_year"].unique()
df_range = df[df["season"].isin(seasons_selected)].copy()

avg_sectors = compute_average_sector(df_range)
overall_avg = yearly_range["Qt (kg/m)"].mean()

st.subheader("Wind rose for selected year range")
fig2 = plot_rose(avg_sectors, overall_avg)
st.plotly_chart(fig2, use_container_width=True)
