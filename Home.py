import streamlit as st
import json
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events


st.set_page_config(page_title="IND320")

st.title("Home")
st.write(
    " Group Project for IND320 - Data til beslutning. "
)

st.sidebar.header("Explorative")
st.sidebar.page_link("pages/01_Data_Table.py", label="Page 1 — Data Table")
st.sidebar.page_link("pages/02_MongoDb.py", label="Page 2 — MongoDB")
st.sidebar.page_link("pages/03_Plot.py", label="Page 3 — Plot")

st.sidebar.header("Anomalies")
st.sidebar.page_link("pages/04_SPC & LOF.py", label="Page 4 — SPC & LOF")

st.sidebar.header("Anomalies")
st.sidebar.page_link("pages/05_STL & Spectogram.py", label="Page 5 — STL & Spectrogram")

# ---------------------------------------------------
# Load GeoJSON
# ---------------------------------------------------
with open("file.geojson", "r") as f:
    geojson_data = json.load(f)


# Initialize selected region
if "selected_area" not in st.session_state:
    st.session_state.selected_area = "NO1"

st.title("Click a Price Area – Highlight Selected Region")


# ---------------------------------------------------
# Build Plotly Figure
# ---------------------------------------------------
fig = go.Figure()
trace_to_area = {}

for idx, feature in enumerate(geojson_data["features"]):
    area = feature["properties"]["ElSpotOmr"]

    coords = feature["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # Highlight selected region
    if st.session_state.selected_area == area:
        line_color = "red"
        line_width = 4
    else:
        line_color = "blue"
        line_width = 1

    fig.add_trace(go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode="lines",
        name=area,
        line=dict(color=line_color, width=line_width)
    ))

    # Map trace number to region name
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

if clicked:
    trace_idx = clicked[0]["curveNumber"]
    region = trace_to_area[trace_idx]

    st.session_state.selected_area = region
    st.success(f"Selected region: **{region}**")

# ---------------------------------------------------
# Display selected region
# ---------------------------------------------------
st.subheader("Currently Selected Region")
st.write(st.session_state.selected_area)
