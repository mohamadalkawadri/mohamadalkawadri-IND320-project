import streamlit as st
import json
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from sidebar import setup_sidebar
from bson.son import SON
from pymongo.mongo_client import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

setup_sidebar()

st.set_page_config(page_title="IND320")

st.title("Visualizing Average Energy Production/Consumption by Price Area")

@st.cache_resource(show_spinner=False)
def get_mongo_collection():
    uri = st.secrets.get("MONGO_URI")
    client = MongoClient(uri, tlsAllowInvalidCertificates=True)
    database = client['assignment4']
    collection = database['elhub']
    return collection

coll = get_mongo_collection()

@st.cache_data(show_spinner=False)
def get_production_groups():
    pipeline = [
        {"$group": {"_id": "$productionGroup"}},
        {"$sort": SON([("_id", 1)])}
    ]
    return [d["_id"] for d in coll.aggregate(pipeline)]

@st.cache_data(show_spinner=False)
def load_data(group: str, year:int, days: int):
    start_date = datetime(year, 1, 1)
    end_date = start_date + timedelta(days=days)

    query = {
        "productionGroup": group,
        "startTime": {
            "$gte": start_date,
            "$lte": end_date
        }
    }

    cursor = coll.find(query, {
        "priceArea": 1,
        "quantityKwh": 1,
        "startTime": 1,
        "_id": 0
    })

    df = pd.DataFrame(list(cursor))
    if not df.empty:
        df["startTime"] = pd.to_datetime(df["startTime"])
    return df

# ---------------------------------------------------
# Load GeoJSON
# ---------------------------------------------------
with open("file.geojson", "r") as f:
    geojson_data = json.load(f)


# Initialize selected region
if "selected_area" not in st.session_state:
    st.session_state.selected_area = "NO1"

if "year" not in st.session_state:
    st.session_state.year = 2021

# --- UI: Choose year ---
st.session_state.year = st.slider(
    "Select year",
    min_value=2021,
    max_value=2024,
    value=st.session_state.year,
    step=1
)

# --- UI: Choose group dynamically based on area ---

groups = get_production_groups()

if groups:
    selected_group = st.selectbox(
        "Choose production/consumption group:",
        groups,
        index=0
    )
else:
    st.warning("No groups found for this area.")
    st.stop()

# --- UI: Choose time interval (days) ---
days = st.slider(
    "Select time interval (days):",
    min_value=1,
    max_value=365,
    value=30,          # default 30 days
    step=1
)

# Functions to calculate the colors based on average quantityKwh

df = load_data(selected_group, st.session_state.year, days)
price_areas = df["priceArea"].unique().tolist()
mean_by_area = (
    df.groupby("priceArea")["quantityKwh"]
    .mean()
    .reindex(price_areas)      # ensure correct order
)
# Normalize values 0–1
vals = mean_by_area.values.astype(float)
minv, maxv = np.nanmin(vals), np.nanmax(vals)
norm = (vals - minv) / (maxv - minv + 1e-9)

# Convert to rgba
def rgba(v, alpha=0.4):
    # blue → red gradient
    r = int(255 * v)
    b = int(255 * (1-v))
    return f"rgba({r},0,{b},{alpha})"

area_colors = {area: rgba(v) for area, v in zip(price_areas, norm)}

# ---------------------------------------------------
# Build Plotly Figure
# ---------------------------------------------------
st.write("Click a Price Area – Highlight Selected Region")
fig = go.Figure()
trace_to_area = {}

for idx, feature in enumerate(geojson_data["features"]):
    area = feature["properties"]["ElSpotOmr"]

    coords = feature["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # Highlight selected region
    if st.session_state.get("selected_area") == area:
        line_color = "red"
        line_width = 4
    else:
        line_color = "blue"
        line_width = 1

    fig.add_trace(go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode="lines",
        fill="toself",
        fillcolor=area_colors.get(area, "rgba(0,0,0,0)"),
        name=f"{area} ({mean_by_area[area]:,.0f} kWh)",
        line=dict(color=line_color, width=line_width)
    ))

    trace_to_area[idx] = area


fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=64.5, lon=15.5),
        zoom=3.8
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

# ---------------------------------------------------
# Capture click event
# ---------------------------------------------------
clicked = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_width="100%",
    override_height=650,
)

# ---------------------------------------------------
# Fix: Extract coords from trace and save in session state
# ---------------------------------------------------
if clicked:
    trace_idx = clicked[0]["curveNumber"]
    point_idx = clicked[0]["pointIndex"]

    # Region name
    region = trace_to_area[trace_idx]
    st.session_state.selected_area = region

    # Extract coordinate from clicked polygon vertex
    lon = fig.data[trace_idx].lon[point_idx]
    lat = fig.data[trace_idx].lat[point_idx]

    st.session_state.selected_coord = {
        "lat": float(lat),
        "lon": float(lon)
    }

    st.success(f"Selected region: **{region}**")
    st.info(f"Saved coord: lat={lat:.5f}, lon={lon:.5f}")
