import streamlit as st

def setup_sidebar():
    st.sidebar.header("Explorative")
    st.sidebar.page_link("pages/01_Data_Table.py", label="Data Table")
    st.sidebar.page_link("pages/02_MongoDb.py", label="MongoDB")
    st.sidebar.page_link("pages/03_Plot.py", label="Plot")

    st.sidebar.header("Anomalies")
    st.sidebar.page_link("pages/04_SPC & LOF.py", label="SPC & LOF")

    st.sidebar.header("Time-Series Analysis")
    st.sidebar.page_link("pages/05_STL & Spectogram.py", label="STL & Spectrogram")
    st.sidebar.page_link("pages/06_Snow Drift.py", label="Snow Drift Analysis")

    st.sidebar.header("Correlation")
    st.sidebar.page_link("pages/07_Sliding_Window_Correlation.py", label="Sliding Window Correlation")

    st.sidebar.header("Prediction")
    st.sidebar.page_link("pages/08_SARIMAX.py", label="SARIMAX Forecast")
