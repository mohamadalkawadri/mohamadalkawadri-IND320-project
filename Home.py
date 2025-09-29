import streamlit as st

st.set_page_config(page_title="IND320")

st.title("Home")
st.write(
    " Group Project for IND320 - Data til beslutning. "
)

st.sidebar.header("Navigation")
st.sidebar.page_link("pages/02_Data_Table.py", label="Page 2 — Data Table")
st.sidebar.page_link("pages/03_Plot.py", label="Page 3 — Plot")

