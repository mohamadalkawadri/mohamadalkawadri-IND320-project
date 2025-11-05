import streamlit as st

st.set_page_config(page_title="IND320")

st.title("Home")
st.write(
    " Group Project for IND320 - Data til beslutning. "
)

st.sidebar.header("Navigation")
st.sidebar.page_link("pages/02_MongoDb.py", label="Page 1 — MongoDB")
st.sidebar.page_link("pages/03_STL_Spectogram.py", label="Page 2 — STL & Spectrogram")
st.sidebar.page_link("pages/04_Data_Table.py", label="Page 3 — Data Table")
st.sidebar.page_link("pages/05_SPC_LOF.py", label="Page 4 — SPC & LOF")
st.sidebar.page_link("pages/06_Plot.py", label="Page 5 — Plot")

