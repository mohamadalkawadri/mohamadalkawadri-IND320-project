import streamlit as st
import pandas as pd
from utils import download_era5_hourly

st.set_page_config(page_title="Data Table")

st.title("Data Table ")

year = st.slider(
    "Select year",
    min_value=2021,
    max_value=2024,
    value=2021,
    step=1
)

@st.cache_data
def load_data(year):
    df = download_era5_hourly(60.3913, 5.3221, year=year)
    return df
df = load_data(year)


rows = []
for col in df.select_dtypes(include="number").columns.tolist():
    series = df[col][:31*24].tolist()
    rows.append(
        {
            "Variable": col,
            "First Month Trend": series,
        }
    )

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    column_config={
        "Variable": st.column_config.TextColumn(),
        "First Month Trend": st.column_config.LineChartColumn(
            label="First Month Trend",
        ),
    },
)
