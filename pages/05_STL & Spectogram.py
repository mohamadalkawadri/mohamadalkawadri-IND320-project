import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from statsmodels.tsa.seasonal import STL
from utils import plot_stl_decomposition, plot_production_spectrogram

st.set_page_config(page_title="STL & Spectrogram")

from sidebar import setup_sidebar

setup_sidebar()
# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('norway_energy_2021.csv')
    return df

df = load_data()

selected_area = st.session_state.selected_area if "selected_area" in st.session_state else "NO1"

st.title("🌤️ Weather Time-Series Analysis — STL & Spectrogram")

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["📈 STL Decomposition", "🔊 Spectrogram"])


with tab1:
    st.header("STL Decomposition (Seasonal-Trend Decomposition using LOESS)")

    # --- UI CONTROLS ---
    period = st.number_input("Seasonal period (hours)", min_value=2, value=24*7, step=1)
    seasonal_smoother = st.slider("Seasonal smoother", min_value=3, max_value=101, value=13, step=2)
    trend_smoother = st.slider("Trend smoother", min_value=21, max_value=1001, value=301, step=20)
    robust = st.checkbox("Robust decomposition", value=True)

    # --- RUN STL ---
    plt = plot_stl_decomposition(
        df,
        area=selected_area,
        group="hydro",
        period=period,
        seasonal_smoother=seasonal_smoother,
        trend_smoother=trend_smoother,
        robust=robust
    )


with tab2:
    st.header("Spectrogram Analysis (Frequency Spectrum Over Time)")
    plot_production_spectrogram(df, area=selected_area, group="hydro")
