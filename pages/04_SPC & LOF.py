import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_temperature_with_spc, plot_precipitation_with_lof, download_era5_hourly
from sidebar import setup_sidebar

setup_sidebar()
# --- PAGE CONFIG ---
st.set_page_config(page_title="Weather Time-Series Analysis", layout="wide")

# --- LOAD DATA ---

coords = st.session_state.get("selected_coord", { "lat": 60.3913, "lon": 5.3221 })
lat = coords.get("lat")
lon = coords.get("lon")

@st.cache_data
def load_data():
    df =  download_era5_hourly(lat, lon, st.session_state.year)
    return df

df = load_data()

selected_area = st.session_state.selected_area if "selected_area" in st.session_state else "NO1"

st.title("🌤️ Weather Time-Series Analysis — SPC & LOF")

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["📈 SPC", "🔊 LOF"])


with tab1:
    st.header("SPC")
    plot_temperature_with_spc(df, city="Bergen")


with tab2:
    st.header("LOF")
    plot_precipitation_with_lof(df, city="Bergen")
