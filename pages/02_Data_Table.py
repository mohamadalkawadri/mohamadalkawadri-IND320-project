import streamlit as st
import pandas as pd
from utils import load_data

st.set_page_config(page_title="Data Table")

st.title("Data Table ")

df = load_data()

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
