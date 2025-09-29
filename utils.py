import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path = "open-meteo-subset.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    return df

def mask_month_range(df: pd.DataFrame, start_month, end_month) -> pd.Series:
    return (df["time"].dt.month >= start_month) & (df["time"].dt.month <= end_month)

