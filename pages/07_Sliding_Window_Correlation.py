import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo.mongo_client import MongoClient
from datetime import datetime

from utils import compute_sliding_correlation, download_era5_hourly
from sidebar import setup_sidebar

# ---------------------------------------------------------------------
# Page & sidebar
# ---------------------------------------------------------------------
st.set_page_config(page_title="Sliding Window Correlation", layout="wide")
setup_sidebar()

# Make sure year exists in session_state (fallback default)
if "year" not in st.session_state:
    st.session_state.year = 2021

year = st.session_state.year

# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_mongo_collection():
    uri = st.secrets.get("MONGO_URI")
    client = MongoClient(uri, tlsAllowInvalidCertificates=True)
    database = client["assignment4"]
    collection = database["elhub"]
    return collection


@st.cache_data(show_spinner=False)
def load_energy(year: int) -> pd.DataFrame:
    """
    Load Elhub energy data from MongoDB for a given year and return a cleaned dataframe.
    Assumes documents have:
        - startTime (datetime or ISO string)
        - priceArea (str)
        - productionGroup (str)
        - quantityKwh (float/int)
    """
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

    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce")
    df = df.dropna(subset=["startTime"])
    df = df.sort_values("startTime")

    return df

coords = st.session_state.get("selected_coord", { "lat": 60.3913, "lon": 5.3221 })
lat = coords.get("lat")
lon = coords.get("lon")

@st.cache_data(show_spinner=False)
def load_meteorology(year: int) -> pd.DataFrame:
    """
    Load ERA5 meteorological data for the given year.
    Adjust the call to `download_era5_hourly` if your signature differs.
    Expected columns in the returned dataframe:
        - time
        - temperature_2m (°C)
        - precipitation (mm)
        - wind_speed_10m (m/s)
        - wind_gusts_10m (m/s)
        - wind_direction_10m (°)
    """

    # Example: change this if your utils.download_era5_hourly has a different API.
    # Common patterns are e.g. (lat, lon, year) or (lat, lon, start, end).
    df_met = download_era5_hourly(lat, lon, year=year)

    df_met["time"] = pd.to_datetime(df_met["time"], utc=True).dt.tz_localize(None)
    df_met = df_met.dropna(subset=["time"])
    df_met = df_met.sort_values("time")

    return df_met


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
energy_df = load_energy(year)
met_df = load_meteorology(year)

# Rename startTime → time to align with meteorology
energy_df = energy_df.rename(columns={"startTime": "time"})

# ---------------------------------------------------------------------
# UI: Title & description
# ---------------------------------------------------------------------
st.title("Sliding Window Correlation")
st.caption(
    "Explore time-varying correlations between a meteorological variable and an "
    "energy production/consumption group with adjustable lag (hours) and window length."
)

# ---------------------------------------------------------------------
# UI: Selectors
#   - price area
#   - energy group
#   - meteorological variable
#   - window & lag
# ---------------------------------------------------------------------
price_areas = sorted(energy_df["priceArea"].unique().tolist())
groups = sorted(energy_df["productionGroup"].unique().tolist())

met_vars = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
]

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    selected_area = st.selectbox("Price area", price_areas, index=0)

with col2:
    energy_group = st.selectbox(
        "Energy production/consumption group",
        groups,
        index=0,
    )

with col3:
    selected_met = st.selectbox("Meteorological variable", met_vars, index=0)

col4, col5 = st.columns([1, 1])

with col4:
    window_hours = st.slider(
        "Window length (hours)",
        min_value=12,
        max_value=720,
        value=168,
        step=12,
        help="Rolling window size used for the correlation calculation.",
    )

with col5:
    lag_hours = st.slider(
        "Lag (hours): meteorology relative to energy",
        min_value=-240,
        max_value=240,
        value=0,
        step=1,
        help=(
            "Positive values shift the meteorological series forward in time "
            "relative to energy; negative values shift it backward."
        ),
    )

# ---------------------------------------------------------------------
# Filter & merge data
# ---------------------------------------------------------------------
# Filter energy for selected price area & group
area_df = energy_df[
    (energy_df["priceArea"] == selected_area)
    & (energy_df["productionGroup"] == energy_group)
].copy()

if area_df.empty:
    st.error(f"No energy data for area={selected_area} and group={energy_group}.")
    st.stop()

# Aggregate in case of duplicate timestamps
energy_series = (
    area_df[["time", "quantityKwh"]]
    .groupby("time", as_index=False)
    .mean()
)

# Keep only needed meteorological column
if selected_met not in met_df.columns:
    st.error(f"Selected meteorological variable '{selected_met}' not found in ERA5 data.")
    st.stop()

met_series = met_df[["time", selected_met]].copy()

# Merge on time
merged = pd.merge(
    energy_series,
    met_series,
    on="time",
    how="inner",
)

if merged.empty:
    st.error("No overlapping timestamps between energy and meteorological data.")
    st.stop()

merged = merged.sort_values("time")
merged = merged.set_index("time")

# Build the series used for correlation:
#   - Energy:    "quantityKwh"
#   - Met var:   selected_met
series = merged[["quantityKwh", selected_met]].dropna()

if series.empty:
    st.error("Not enough overlapping data points between the selected series.")
    st.stop()

# ---------------------------------------------------------------------
# Sliding window correlation
#   We treat:
#       X = energy (quantityKwh)
#       Y = meteorological variable (selected_met), which is lagged
# ---------------------------------------------------------------------
result_df, overall_corr = compute_sliding_correlation(
    series,
    x_col="quantityKwh",
    y_col=selected_met,
    window=window_hours,
    lag=lag_hours,
)

# For plotting: original energy + lagged meteorology
result_df["Energy (Series A)"] = series["quantityKwh"]
result_df["Meteorology (Series B, lagged)"] = series[selected_met].shift(lag_hours)

# Peak rolling correlation
valid_corr = result_df["rolling_corr"].dropna()
peak_time = None
peak_corr = None
if not valid_corr.empty:
    peak_idx = valid_corr.idxmax()
    peak_time = peak_idx
    peak_corr = valid_corr.loc[peak_idx]

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
metric_cols = st.columns(3)
metric_cols[0].metric("Overall corr (with lag)", f"{overall_corr:.3f}")
metric_cols[1].metric(
    "Max rolling corr",
    f"{peak_corr:.3f}" if peak_corr is not None else "n/a",
    help=f"Peak at {peak_time}" if peak_time is not None else None,
)
metric_cols[2].metric("Window (hours)", window_hours)

# ---------------------------------------------------------------------
# Plotly figure
#   Top: Energy & Meteorology (lagged)
#   Bottom: rolling correlation
# ---------------------------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.55, 0.45],
)

time_index = result_df.index

# Top: Energy series
fig.add_trace(
    go.Scatter(
        x=time_index,
        y=result_df["Energy (Series A)"],
        mode="lines",
        name=f"Energy: {energy_group}",
        line=dict(color="steelblue"),
    ),
    row=1,
    col=1,
)

# Top: Meteorological series (lagged)
fig.add_trace(
    go.Scatter(
        x=time_index,
        y=result_df["Meteorology (Series B, lagged)"],
        mode="lines",
        name=f"Met: {selected_met} (lagged {lag_hours}h)",
        line=dict(color="crimson"),
    ),
    row=1,
    col=1,
)

# Bottom: Rolling correlation
fig.add_trace(
    go.Scatter(
        x=time_index,
        y=result_df["rolling_corr"],
        mode="lines",
        name="Rolling correlation",
        line=dict(color="darkgreen"),
    ),
    row=2,
    col=1,
)

fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=2, col=1)

fig.update_layout(
    height=800,
    title=(
        f"{selected_area} — Energy group: {energy_group} vs {selected_met} "
        f"(lag {lag_hours}h, window {window_hours}h, year {year})"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="Quantity (kWh)", row=1, col=1)
fig.update_yaxes(title_text="Rolling correlation", range=[-1.05, 1.05], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------
with st.expander("Show correlation data"):
    preview_cols = [
        "Energy (Series A)",
        "Meteorology (Series B, lagged)",
        "rolling_corr",
    ]
    st.dataframe(
        result_df[preview_cols]
        .rename(columns={"rolling_corr": "rolling_corr (two-sided)"})
        .dropna(),
        use_container_width=True,
    )
