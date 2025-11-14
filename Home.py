import streamlit as st

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
